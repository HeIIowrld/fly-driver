"""Replay saved cardinal FlyVis edges through trained temporal and original flight paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "source", "horizontal", "vertical", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    training, candidate, source, horizontal, vertical = map(
        Path, (a.training, a.candidate, a.source, a.horizontal, a.vertical)
    )
    tr, tm, cr, sr = (
        read(training / "result.json"),
        read(training / "manifest.json"),
        read(candidate / "result.json"),
        read(source / "result.json"),
    )
    plan = tm["config"]["plan"]
    root = Path(plan["data"])
    prior_model = read(Path(plan["source"]) / "result.json")
    module = "src/flybrain_drive/flight_temporal_motion_boundary.py"
    prior_records = []
    for path, status in (
        (horizontal, "completed_centered_boundary_saved_edge_comparison"),
        (vertical, "completed_centered_boundary_saved_orthogonal_edge_comparison"),
    ):
        result, config = read(path / "result.json"), read(path / "manifest.json")["config"]
        if (
            result["status"] != status
            or sha256(path / "replay.npz") != result["replay_sha256"]
            or config["source_responses_sha256"] != sr["responses_sha256"]
        ):
            raise ValueError("Exact previously saved directional replay required")
        prior_records.append((path, result, config))
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(prior_model["checkpoint"]) != prior_model["checkpoint_sha256"]
        or cr["status"] != "completed_temporal_motion_boundary_onepass"
        or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or cr["boundary_source_sha256"] != sha256(module)
        or sr["status"] != "completed"
        or sha256(source / "responses.npz") != sr["responses_sha256"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact trained temporal source, graph and FlyVis responses required")
    stimulus = read(source / "stimulus.json")
    if stimulus["mode"] != "edge" or stimulus["dt"] != 0.005:
        raise ValueError("Saved 5ms FlyVis edge responses required")
    pairs = [(contrast, angle) for contrast in (0, 1) for angle in (0, 90, 180, 270)]
    selected = []
    for contrast, angle in pairs:
        indices = [
            i
            for i, row in enumerate(stimulus["parameters"])
            if row["intensity"] == contrast and row["angle"] == angle and row["speed"] == 19.0
        ]
        if len(indices) != 1:
            raise ValueError("Each declared saved edge must exist exactly once")
        selected.append(indices[0])
    with load_array(source / "responses.npz", allow_pickle=False) as x:
        names = x["central_cell_types"].tolist()
        indices = [
            names.index(f"{kind}{direction}") for kind in ("T4", "T5") for direction in "abcd"
        ]
        length = min(int(x["valid_frames"][i]) for i in selected)
        activity = x["activity"][selected, :length, :][:, :, indices].copy()
    if length < 2 or not np.isfinite(activity).all():
        raise ValueError("Finite saved edge histories required")
    entries = [e for e in read(root / "dataset.json")["episodes"] if e["role"] == "train"]
    if len(entries) != 6 or sha256(root / entries[0]["file"]) != entries[0]["sha256"]:
        raise ValueError("Original first TRAIN body history required")
    names = ("visual", "visual_valid", "body", "body_valid", "goals")
    with load_array(root / entries[0]["file"], allow_pickle=False) as x:
        data = {key: x[key][:1].copy() for key in names}
    if data["visual_valid"].any():
        raise ValueError("Original visual6 mask must remain invalid")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"],
        torch.load(prior_model["checkpoint"], map_location="cpu", weights_only=True),
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original v7 graph contract required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    temporal = TemporalMotionBoundary(
        model.backbone.scene_command.mean,
        model.backbone.scene_command.scale,
        len(model.backbone.vision_ports),
    ).to("cuda")
    weight = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    if (
        weight["schema"] != "flybrain.temporal_motion_boundary.v1"
        or weight["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or weight["actor_qualified"] is not False
        or weight["time_constants_seconds"] != [0.05, 0.2]
    ):
        raise ValueError("Explicit unqualified temporal candidate required")
    temporal.load_state_dict(weight["model"], strict=True)
    temporal.eval().requires_grad_(False)
    if not torch.equal(temporal.mean, model.backbone.scene_command.mean) or not torch.equal(
        temporal.scale, model.backbone.scene_command.scale
    ):
        raise ValueError("Original TRAIN normalization required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "checkpoint_sha256": tr["checkpoint_sha256"],
            "temporal_checkpoint_sha256": cr["checkpoint_sha256"],
            "source_response_sha256": sr["responses_sha256"],
            "source_stimulus_sha256": sha256(source / "stimulus.json"),
            "prior_replay_hashes": [item[1]["replay_sha256"] for item in prior_records],
            "body_episode_sha256": entries[0]["sha256"],
            "dt_seconds": 0.005,
            "directions": [0, 90, 180, 270],
            "contrasts": [0, 1],
            "synthetic_spatial_replication": "central T4/T5 response repeated in 16 bins per view",
            "direct_motion": "steering/goal ports fixed to TRAIN mean motion",
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    t = {
        key: torch.tensor(value, device="cuda").expand(33, *value.shape[1:])
        for key, value in data.items()
    }
    mean = model.backbone.scene_command.mean
    direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    state, traces, held = model.initial_state(66), None, None
    outputs, deltas = [], []
    with torch.no_grad():
        for frame in range(length):
            if frame:
                for dt in substeps(0.005):
                    state = model._advance_elapsed_tensor(held, state, dt)
            outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
            scene = mean.expand(33, -1).clone().reshape(33, 4, 9, 16)
            for condition in range(8):
                for view in range(4):
                    scene[condition * 4 + view, view, 1:, :] = torch.tensor(
                        np.tanh(activity[condition, frame]), device="cuda"
                    )[:, None]
            scene = scene.flatten(1)
            base = model._drive_tensor(*(t[key] for key in names), scene)
            reference = model._drive_tensor(*(t[key] for key in names), mean.expand(33, -1))
            base[:, direct] = reference[:, direct]
            delta, traces = temporal.step(
                scene,
                torch.ones((33, 4), dtype=torch.bool, device="cuda"),
                traces,
                None if frame == 0 else 0.005,
            )
            changed = temporal.modulate(base, model.backbone.vision_ports, delta)
            held = torch.cat([base, changed])
            deltas.append(delta.cpu().numpy())
    output, delta = np.stack(outputs), np.stack(deltas)
    errors = []
    for prior_index, (path, _, _) in enumerate(prior_records):
        with load_array(path / "replay.npz", allow_pickle=False) as previous:
            old = previous["outputs"]
            for contrast in range(2):
                for pair_position in range(2):
                    angle_index = ((0, 2) if prior_index == 0 else (1, 3))[pair_position]
                    for view in range(4):
                        current = contrast * 16 + angle_index * 4 + view
                        before = contrast * 8 + pair_position * 4 + view
                        errors.append(float(np.max(np.abs(output[1:, current] - old[:-1, before]))))
            errors.append(float(np.max(np.abs(output[1:, 32] - old[:-1, 16]))))
    if max(errors) > 1e-6:
        raise ValueError("Original direction response must match earlier fixed replay after 1 step")
    baseline_error = float(np.max(np.abs(output[:, 32] - output[:, 65])))
    if baseline_error > 1e-6:
        raise ValueError("Constant mean motion must keep temporal baseline unchanged")
    summary = {}
    for contrast in range(2):
        for ai, bi in ((0, 2), (1, 3)):
            for view, name in enumerate(("front", "left", "right", "rear")):
                ia, ib = contrast * 16 + ai * 4 + view, contrast * 16 + bi * 4 + view
                before = output[:, ia] - output[:, ib]
                after = output[:, 33 + ia] - output[:, 33 + ib]
                summary[f"{contrast}_{pairs[ia // 4][1]}_{pairs[ib // 4][1]}_{name}"] = {
                    "original_direction_max_abs": np.max(np.abs(before), axis=0).tolist(),
                    "learned_direction_max_abs": np.max(np.abs(after), axis=0).tolist(),
                    "direction_difference_change_max_abs": np.max(
                        np.abs(after - before), axis=0
                    ).tolist(),
                }
    np.savez_compressed(
        run / "replay.npz", output=output, temporal_port_delta=delta, central_activity=activity
    )
    result = {
        "status": "completed_temporal_flyvis_saved_direction_replay",
        "frames": length,
        "original_prior_max_error": max(errors),
        "mean_motion_baseline_max_error": baseline_error,
        "temporal_port_delta_max_abs": float(np.max(np.abs(delta))),
        "conditions": summary,
        "replay_sha256": sha256(run / "replay.npz"),
        "central_spatial_replication_synthetic": True,
        "temporal_weights_learned_from_driving_not_fly_physiology": True,
        "real_driving_verified": False,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
