"""Train-only v6 diagnosis: isolated Adam steps and causal stale/refreshed state controls.

No fit, checkpoint selection, deployment or biological inference. Original histories are
replayed from their own starts; the saved training/development split is never changed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_learning import bounded_intent_loss
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import drives_for, substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


@torch.no_grad()
def boundary(model, data, schedule, start):
    """State just before advancing from start-1 to start; keep the original held input."""
    state, held = model.initial_state(1), None
    if start:
        drive = drives_for(model, data, 0, start)
        for index in range(start):
            if index:
                for dt in schedule[index]:
                    state = model._advance_elapsed_tensor(held, state, dt)
            held = drive[index:index + 1]
    return state.detach(), None if held is None else held.detach()


def window(model, data, schedule, start, stop, state, held):
    drive = drives_for(model, data, start, stop)
    outputs, selected = [], []
    for local, index in enumerate(range(start, stop)):
        if index:
            for dt in schedule[index]:
                state = model._advance_elapsed_tensor(held, state, dt)
        if data["target_valid"][index, 0]:
            outputs.append(model._read_state_preactivation_tensor(state)[0][0])
            selected.append(index)
        held = drive[local:local + 1]
    prediction = torch.stack(outputs)
    target = data["target"][selected]
    valid = data["target_valid"][selected].clone()
    valid[:, 1] = False
    loss = bounded_intent_loss(prediction, target, valid, torch.tensor([.1, .1], device="cuda"))
    report = dict(
        labels=len(selected), loss=float(loss.detach()),
        turn_rmse=float((prediction[:, 0].tanh() - target[:, 0]).square().mean().sqrt().detach()),
        turn_min=float(prediction[:, 0].tanh().min().detach()),
        turn_max=float(prediction[:, 0].tanh().max().detach()),
    )
    return loss, report


@torch.no_grad()
def command_summary(model, data):
    module = model.scene_command
    x = ((data["scene_features"] - module.mean) / module.scale).clamp(-5, 5)
    logits = module.linear(x)
    current = logits.sigmoid()
    return dict(
        normalized_clip_fraction=float((x.abs() >= 5).float().mean()),
        logit_min=logits.min(0).values.cpu().tolist(),
        logit_max=logits.max(0).values.cpu().tolist(),
        current_min=current.min(0).values.cpu().tolist(),
        current_max=current.max(0).values.cpu().tolist(),
        current_mean=current.mean(0).cpu().tolist(),
        current_over_point1_fraction=(current > .1).float().mean(0).cpu().tolist(),
        current_under_point0003_fraction=(current < .0003).float().mean(0).cpu().tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    plan, seed = config["plan"], 863152
    data_root = Path(plan["data"])
    parent_receipt = read(Path(plan["source"]) / "result.json")
    if (sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
            or sha256(parent_receipt["checkpoint"]) != config["source_checkpoint_sha256"]
            or sha256(plan["spec"]) != config["spec_sha256"]
            or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]):
        raise ValueError("Original checkpoints, spec and data manifest required")
    fractions = [0., .001, .01, .1, 1.]
    run = begin_run(args.output, dict(
        arguments=vars(args), seed=seed, scope=__doc__,
        training_manifest_sha256=sha256(root / "manifest.json"),
        checkpoint_sha256=receipt["checkpoint_sha256"],
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        step_fractions=fractions, learning_rate=plan["learning_rate"],
        windows="first and last label-bearing original 64-observation chunks per train episode",
        optimizer="fresh Adam per independent probe, not original sequential optimizer state",
        development_inputs_used=False, development_labels_used=False,
        source_recordings_aid_absence_verified=False,
        actor_qualified=False, actuator_messages_sent=0,
    ))
    result = dict(status="failed", actor_qualified=False, goal_achieved=False,
                  actuator_messages_sent=0, persisted_training_checkpoints=0)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        parent = torch.load(parent_receipt["checkpoint"], map_location="cpu", weights_only=True)
        final = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
        model = VisualFlightCircuit(plan["spec"], glutamate_sign=parent["glutamate_sign"])
        model.import_v5(parent)
        model.to("cuda")
        episodes = []
        for entry in read(data_root / "dataset.json")["episodes"]:
            if entry["role"] != "train":
                continue  # Do not even open development episode arrays.
            if sha256(data_root / entry["file"]) != entry["sha256"]:
                raise ValueError("Original training episode changed")
            with load_array(data_root / entry["file"], allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                    if value.dtype.kind in "fbiu" and key != "observation_time"}
            schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
            episodes.append((entry, data, schedule, arrays))
        features = torch.cat([data["scene_features"] for _, data, _, _ in episodes])
        model.scene_command.set_train_normalization(
            features.mean(0),
            features.std(0, correction=0).clamp_min(plan["normalization_std_floor"]))
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        parameters = [p for p in model.parameters() if p.requires_grad]
        if len(parameters) != 2:
            raise ValueError("Only scene weight and bias may be probed")
        summaries = []
        for label, state in [("initial", initial), ("epoch06", final["model"])]:
            model.load_state_dict(state, strict=True)
            for entry, data, _, _ in episodes:
                summaries.append(dict(stage=label, file=entry["file"],
                                      **command_summary(model, data)))
        probes = []
        for entry, data, schedule, arrays in episodes:
            starts = [start for start in range(0, len(schedule), plan["chunk_observations"])
                      if arrays["target_valid"][start:start + plan["chunk_observations"], 0].any()]
            for start in dict.fromkeys([starts[0], starts[-1]]):
                stop = min(start + plan["chunk_observations"], len(schedule))
                model.load_state_dict(initial, strict=True)
                state, held = boundary(model, data, schedule, start)
                optimizer = torch.optim.Adam(parameters, lr=plan["learning_rate"])
                optimizer.zero_grad(set_to_none=True)
                loss, before = window(model, data, schedule, start, stop, state, held)
                if not loss.requires_grad:
                    probes.append(dict(file=entry["file"], start=start, stop=stop,
                                       skipped="no connected current-window target", before=before))
                    continue
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                gradients = [p.grad.detach().clone() for p in parameters]
                values = [p.detach().clone() for p in parameters]
                optimizer.step()
                deltas = [p.detach().clone() - value
                          for p, value in zip(parameters, values, strict=True)]
                dot = sum(float((g * d).sum()) for g, d in zip(gradients, deltas, strict=True))
                controls = []
                for fraction in fractions:
                    with torch.no_grad():
                        for p, value, delta in zip(parameters, values, deltas, strict=True):
                            p.copy_(value + fraction * delta)
                        _, stale = window(model, data, schedule, start, stop, state, held)
                        refreshed_state, refreshed_held = boundary(model, data, schedule, start)
                        _, refreshed = window(model, data, schedule, start, stop,
                                              refreshed_state, refreshed_held)
                    controls.append(dict(fraction=fraction, stale=stale, refreshed=refreshed))
                probes.append(dict(
                    file=entry["file"], start=start, stop=stop,
                    original_start_time=float(arrays["observation_time"][start]),
                    original_stop_time=float(arrays["observation_time"][stop - 1]),
                    before=before, unclipped_gradient_norm=float(norm),
                    clipped_gradient_dot_full_adam_delta=dot,
                    maximum_parameter_step=max(float(d.abs().max()) for d in deltas),
                    controls=controls,
                ))
                write_json(run / "progress.json", dict(probes=len(probes)))
        model.load_state_dict(initial, strict=True)
        for key, value in model.state_dict().items():
            reference = initial[key]
            if value.is_sparse:
                assert torch.equal(value.coalesce().indices(), reference.coalesce().indices())
                assert torch.equal(value.coalesce().values(), reference.coalesce().values())
            else:
                assert torch.equal(value, reference), key
        result.update(status="completed_train_only_diagnostic", probes=probes,
                      command_summaries=summaries, full_initial_state_restored=True,
                      development_inputs_used=False, development_labels_used=False,
                      original_clocks_preserved=True, prefix_replayed_from_original_start=True,
                      source_recordings_aid_absence_verified=False,
                      physical_vehicle_plant_tested=False, physiological_recognition_verified=False,
                      termination_reason="declared independent Adam probes measured; no fit saved")
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / "result.json", result)
        raise
    write_json(run / "result.json", result)
    print(json.dumps(dict(status=result["status"], probes=len(probes))))


if __name__ == "__main__":
    main()
