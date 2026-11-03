"""Minimum-change virtual-body yaw calibration through fixed measured flight edges."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.flight_learning import GOAL_CONTRACT
from flybrain_drive.graph import sha256
from scripts.audit_flight_feedback_paths import encode, fixed_inputs, read, settle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    receipt, configuration = read(root / "result.json"), read(root / "manifest.json")["config"]
    checkpoint, spec = Path(receipt["checkpoint"]), configuration["plan"]["spec"]
    data_root = Path(configuration["arguments"]["data"])
    if (sha256(checkpoint) != receipt["checkpoint_sha256"]
            or sha256(spec) != configuration["spec_sha256"]
            or sha256(data_root / "dataset.json") != configuration["dataset_manifest_sha256"]):
        raise ValueError("Unchanged model/spec/source dataset required")
    dataset = read(data_root / "dataset.json")
    entry = dataset["episodes"][0]
    if entry["role"] != "train" or sha256(data_root / entry["file"]) != entry["sha256"]:
        raise ValueError("One explicitly training-only calibration snapshot required")
    with load_array(data_root / entry["file"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    run = begin_run(args.output, dict(arguments=vars(args), seed=863120,
        source_checkpoint_sha256=sha256(checkpoint), spec_sha256=sha256(spec),
        dataset_manifest_sha256=sha256(data_root / "dataset.json"),
        target_steady_left_intent_per_yaw=-.1,
        target_basis="engineering yaw damping, not fly tuning",
        calibration_scene=entry["file"], calibration_index=0,
        artificial_zero_tilt_and_rotation=True,
        model_neutral_steps=200, dt=.01, estimated_receptor_tuning=False,
        parameters_allowed_to_change="body.weight[:,2] only; gyro yaw channel",
        method="minimum Euclidean yaw-column update under exact local graph Jacobian",
        teacher_targets_used=False, heldout_inputs_used=False, actuator_messages_sent=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    model = FlightCircuit(spec, glutamate_sign=configuration["plan"]["glutamate_sign"]).to("cuda")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["goal_contract"] != GOAL_CONTRACT:
        raise ValueError("Explicit v4 task semantics required")
    original_matrix = model.circuit.matrix.coalesce().detach().clone()
    model.load_state_dict(payload["model"], strict=True)
    if (not torch.equal(original_matrix.indices(), model.circuit.matrix.coalesce().indices())
            or not torch.equal(original_matrix.values(), model.circuit.matrix.coalesce().values())):
        raise ValueError("Original graph differs from checkpoint")
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    inputs = fixed_inputs(arrays, 1)
    drive = encode(model, inputs).detach().requires_grad_(True)
    state = settle(model, drive)
    turn = model._read_state_tensor(state)[0][0, 0]
    port_gradient = torch.autograd.grad(turn, drive)[0][0, model.body_ports]
    rate = drive.detach()[0, model.body_ports]
    sensitivity = (port_gradient * rate * (1 - rate)).detach()
    old_yaw = model.body.weight[:, 2].detach().clone()
    previous = sensitivity @ old_yaw
    norm = sensitivity.square().sum()
    if float(norm) < 1e-8:
        raise ValueError("Insufficient physical graph sensitivity; cannot calibrate a bypass")
    correction = ((-.1 - previous) / norm) * sensitivity
    with torch.no_grad():
        model.body.weight[:, 2].add_(correction)
    measured = sensitivity @ model.body.weight[:, 2]
    if not torch.isclose(measured, torch.tensor(-.1, device="cuda"), atol=1e-6):
        raise ValueError("Minimum-change projection failed its local Jacobian constraint")
    changed = []
    for key, value in model.state_dict().items():
        if key == "body.weight":
            torch.testing.assert_close(value[:, [0, 1, 3, 4, 5]],
                                       before[key][:, [0, 1, 3, 4, 5]], rtol=0, atol=0)
        elif value.is_sparse:
            if (not torch.equal(value.coalesce().indices(), before[key].coalesce().indices())
                    or not torch.equal(value.coalesce().values(), before[key].coalesce().values())):
                raise ValueError("Fixed sparse graph changed")
        elif not torch.equal(value, before[key]):
            raise ValueError("Unrelated boundary/readout/buffer changed: " + key)
        if key == "body.weight":
            changed.append(key + "[:,2]")
    saved = dict(payload)
    saved.update(schema="flybrain.flight.speed_yaw_calibration.v1", model=model.state_dict(),
                 source_checkpoint_sha256=sha256(checkpoint), actor_qualified=False,
                 yaw_boundary_calibration=dict(target_derivative=-.1,
                     calibration_scene=entry["file"], index=0, physiology_verified=False,
                     freeze_against_unconstrained_imitation=True))
    torch.save(saved, run / "checkpoint.pt")
    np.savez_compressed(run / "yaw-column-projection.npz", old=old_yaw.cpu().numpy(),
                        correction=correction.cpu().numpy(),
                        sensitivity=sensitivity.cpu().numpy(),
                        new=model.body.weight[:, 2].detach().cpu().numpy())
    write_json(run / "result.json", dict(status="completed_model_calibration",
        checkpoint=str(run / "checkpoint.pt"), checkpoint_sha256=sha256(run / "checkpoint.pt"),
        previous_local_derivative=float(previous), target_local_derivative=-.1,
        projected_local_derivative=float(measured.detach()), correction_l2=float(correction.norm()),
        maximum_absolute_new_yaw_weight=float(model.body.weight[:, 2].abs().max().detach()),
        changed_parameters=changed, original_graph_unchanged=True,
        all_other_parameters_and_buffers_unchanged=True, teacher_targets_used=False,
        heldout_inputs_used=False, physiological_stabilization_verified=False,
        independent_driving_validated=False, actor_qualified=False, actuator_messages_sent=0,
        source_training=str(root), goal_achieved=False,
        termination_reason="one train-snapshot engineering Jacobian constraint projected exactly"))
    print(json.dumps(dict(previous=float(previous), projected=float(measured.detach()),
                         correction_norm=float(correction.norm()))))


if __name__ == "__main__":
    main()
