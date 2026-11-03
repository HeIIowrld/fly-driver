"""Measure fixed-graph visual-boundary gains; do not equate gradients with recognition."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT, SeparatedFlightCircuit
from flybrain_drive.graph import sha256
from scripts.audit_flight_feedback_paths import encode, read, settle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, data_root = Path(args.candidate), Path(args.data)
    receipt, configuration = read(root / "result.json"), read(root / "manifest.json")["config"]
    checkpoint, spec = receipt["checkpoint"], configuration["arguments"]["spec"]
    dataset = read(data_root / "dataset.json")
    if (sha256(checkpoint) != receipt["checkpoint_sha256"]
            or sha256(spec) != configuration["spec_sha256"]
            or sha256(data_root / "dataset.json") != configuration["dataset_manifest_sha256"]):
        raise ValueError("Unchanged v5 candidate/spec and data required")
    contexts = [(1, 100), (2, 300), (3, 200), (4, 400)]
    run = begin_run(args.output, dict(arguments=vars(args), seed=863140, contexts=contexts,
        checkpoint_sha256=sha256(checkpoint), model_neutral_steps=200, dt=.01,
        input_scope="fixed actual scene/body/task snapshot; model-only steady-state gradients",
        comparison="per-neuron derivative per equal external current unit; not physiology",
        finite_difference_epsilon=.01, optimizer_updates=0, actuator_messages_sent=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["goal_contract"] != COMMAND_GOAL_CONTRACT:
        raise ValueError("Explicit separated commands required")
    model = SeparatedFlightCircuit(spec, glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.assert_body_boundary_protected()
    reports, gradients, feature_gradients = [], [], []
    for number, index in contexts:
        entry = dataset["episodes"][number]
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Source complete episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            inputs = {key: torch.tensor(archive[key][index:index + 1], device="cuda")
                      for key in ("visual", "visual_valid", "body", "body_valid", "goals",
                                  "scene_features")}
        drive = encode(model, inputs).detach().requires_grad_(True)
        intent = model._read_state_tensor(settle(model, drive))[0][0]
        gradient = torch.stack([torch.autograd.grad(intent[axis], drive,
                                retain_graph=axis == 0)[0][0] for axis in range(2)])
        rate = drive.detach()[0, model.vision_ports]
        scene_gradient = ((gradient[:, model.vision_ports] * rate * (1 - rate))
                          @ model.vision.weight[:, 12:]).detach()
        group_reports = {}
        for name, ports in [("HSVS", model.vision_ports), ("steering_DN", model.steering_ports),
                             ("power_DN", model.goal_ports), ("haltere", model.body_ports)]:
            values = gradient[:, ports].abs()
            group_reports[name] = dict(count=len(ports),
                median_abs=values.median(1).values.cpu().tolist(),
                maximum_abs=values.max(1).values.cpu().tolist())
        finite = []
        for name, ports in [("HSVS", model.vision_ports), ("steering_DN", model.steering_ports)]:
            selected = int(ports[gradient[0, ports].abs().argmax()])
            with torch.no_grad():
                perturbed = drive.detach().repeat(2, 1)
                perturbed[0, selected] -= .01
                perturbed[1, selected] += .01
                output = model._read_state_tensor(settle(model, perturbed))[0][:, 0]
                difference = float((output[1] - output[0]) / .02)
            finite.append(dict(group=name, graph_index=selected,
                               autograd=float(gradient[0, selected]),
                               central_difference=difference))
        reports.append(dict(file=entry["file"], index=index, role=entry["role"],
                            speed_kmh=entry["desired_speed_kmh"], groups=group_reports,
                            scene_gradient_l2=scene_gradient.norm(dim=1).cpu().tolist(),
                            numerical_checks=finite))
        gradients.append(gradient.detach().cpu().numpy())
        feature_gradients.append(scene_gradient.cpu().numpy())
    np.savez_compressed(run / "boundary-gradients.npz", gradient=np.stack(gradients),
                        scene_gradient=np.stack(feature_gradients))
    write_json(run / "result.json", dict(status="completed", contexts=reports,
        interpretation="model numerical transmission only; useful visual recognition unverified",
        gradients_sha256=sha256(run / "boundary-gradients.npz"), actor_qualified=False,
        optimizer_updates=0, actuator_messages_sent=0, goal_achieved=False,
        termination_reason="equal-unit boundary sensitivity and central differences measured"))
    print(json.dumps(dict(status="completed", contexts=len(reports))))


if __name__ == "__main__":
    main()
