"""Causal fixture for sustained motion on saved central FlyVis edge responses.

The slow projection is a +/- copy of an unrelated transient weight, not a fitted
model or biological mapping. No predictions are used to control a vehicle.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_sustained_opponent_boundary import (
    SustainedOpponentBoundary,
    opponent_axes,
)
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "prior", "output"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    training, candidate, prior = map(Path, (a.training, a.candidate, a.prior))
    tr, cr, pr = (read(p / "result.json") for p in (training, candidate, prior))
    pc = read(prior / "manifest.json")["config"]
    plan = read(training / "manifest.json")["config"]["plan"]
    root = Path(plan["data"])
    source = read(Path(plan["source"]) / "result.json")
    if (sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
            or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
            or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
            or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or pr["status"] != "completed_opponent_temporal_flyvis_saved_direction_replay"
            or sha256(prior / "replay.npz") != pr["replay_sha256"]
            or pc["checkpoint_sha256"] != tr["checkpoint_sha256"]
            or pc["temporal_checkpoint_sha256"] != cr["checkpoint_sha256"]
            or not torch.cuda.is_available()):
        raise ValueError("Original model, temporal checkpoint and prior replay required")
    entries = [e for e in read(root / "dataset.json")["episodes"] if e["role"] == "train"]
    if len(entries) != 6 or sha256(root / entries[0]["file"]) != pc["body_episode_sha256"]:
        raise ValueError("Prior first TRAIN body source required")
    names = ("visual", "visual_valid", "body", "body_valid", "goals")
    with load_array(root / entries[0]["file"], allow_pickle=False) as data:
        body_data = {key: data[key][:1].copy() for key in names}
    with load_array(prior / "replay.npz", allow_pickle=False) as data:
        activity, old = data["central_activity"].copy(), data["output"].copy()
    if activity.ndim != 3 or activity.shape[0] != 8 or activity.shape[2] != 8:
        raise ValueError("Eight saved central T4/T5 stimulus histories required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan["source_spec"], torch.load(
        source["checkpoint"], map_location="cpu", weights_only=True)).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original v7 graph required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    saved = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    boundary = SustainedOpponentBoundary(model.backbone.scene_command.mean,
        model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to("cuda")
    missing, unexpected = boundary.load_state_dict(saved["model"], strict=False)
    if missing != ["sustained_projection.weight"] or unexpected or saved["actor_qualified"]:
        raise ValueError("Only new unqualified sustained projection may be missing")
    boundary.eval().requires_grad_(False)
    run = begin_run(a.output, {"program_sha256": sha256(__file__),
        "boundary_sha256": sha256("src/flybrain_drive/flight_sustained_opponent_boundary.py"),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "temporal_checkpoint_sha256": cr["checkpoint_sha256"],
        "prior_replay_sha256": pr["replay_sha256"],
        "body_episode_sha256": entries[0]["sha256"],
        "conditions": ["original", "transient", "fixture_plus", "fixture_minus"],
        "fixture": "slow weights = +/- original trained transient weights; no new learning",
        "scene": "saved central FlyVis edges copied to 16 bins of one view, synthetic spatial",
        "direct_motion": "steering and goal ports fixed to TRAIN mean motion",
        "labels_opened": False, "actor_deployed": False, "optimizer_updates": 0})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    data = {k: torch.tensor(v, device="cuda").expand(33, *v.shape[1:])
            for k, v in body_data.items()}
    mean = model.backbone.scene_command.mean
    direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    state, traces, held, outputs = model.initial_state(132), None, None, []
    delta_effect = []
    with torch.no_grad():
        weight = boundary.opponent_projection.weight.detach().clone()
        for frame in range(len(activity[0])):
            if frame:
                for dt in substeps(.005):
                    state = model._advance_elapsed_tensor(held, state, dt)
            outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
            scene = mean.expand(33, -1).clone().reshape(33, 4, 9, 16)
            for condition in range(8):
                for view in range(4):
                    scene[condition * 4 + view, view, 1:, :] = torch.tensor(
                        np.tanh(activity[condition, frame]), device="cuda")[:, None]
            scene = scene.flatten(1)
            base = model._drive_tensor(*(data[k] for k in names), scene)
            reference = model._drive_tensor(*(data[k] for k in names), mean.expand(33, -1))
            base[:, direct] = reference[:, direct]
            boundary.sustained_projection.weight.zero_()
            transient, traces = boundary.step(scene, torch.ones((33, 4), dtype=torch.bool,
                device="cuda"), traces, None if frame == 0 else .005)
            # The same current trace is used in each intervention, with no extra time advance.
            slow = opponent_axes(traces[1])
            fast = opponent_axes(traces[0] - traces[1])
            transient_logits = boundary.opponent_projection(fast)
            plus = .5 * torch.tanh(transient_logits + slow @ weight.T)
            minus = .5 * torch.tanh(transient_logits - slow @ weight.T)
            delta_effect.append(torch.stack(((plus-transient).abs().mean(),
                                             (minus-transient).abs().mean())).cpu().numpy())
            held = torch.cat([base] + [boundary.modulate(base, model.backbone.vision_ports, x)
                                       for x in (transient, plus, minus)])
    output, effects = np.stack(outputs), np.stack(delta_effect)
    if old.shape != output[:, :66].shape:
        raise ValueError("Prior replay dimensions changed")
    error = float(np.max(np.abs(output[:, :66] - old)))
    if error > 1e-6:
        raise ValueError(f"Existing original/transient replay mismatch: {error}")
    pairs = [(0, 2), (1, 3)]
    reports = {}
    for contrast in range(2):
        for ai, bi in pairs:
            for view, name in enumerate(("front", "left", "right", "rear")):
                ia, ib = contrast * 16 + ai * 4 + view, contrast * 16 + bi * 4 + view
                differences = [output[:, offset + ia] - output[:, offset + ib]
                               for offset in (0, 33, 66, 99)]
                reports[f"{contrast}_{ai}_{bi}_{name}"] = {
                    "direction_steering_max_abs": [float(np.max(np.abs(d[:, 0])))
                                                   for d in differences],
                    "fixture_change_steering_max_abs": [float(np.max(np.abs(
                        differences[j][:, 0] - differences[1][:, 0]))) for j in (2, 3)]}
    np.savez_compressed(run / "replay.npz", output=output, mean_abs_port_effect=effects)
    write_json(run / "result.json", {"status": "completed_sustained_flyvis_fixture_replay",
        "original_and_transient_max_error": error,
        "mean_abs_port_effect": effects.mean(axis=0).tolist(), "reports": reports,
        "replay_sha256": sha256(run / "replay.npz"), "actor_deployed": False,
        "optimizer_updates": 0})


if __name__ == "__main__":
    main()
