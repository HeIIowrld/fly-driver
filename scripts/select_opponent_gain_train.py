"""Select a fixed opponent projection gain on original 30 km/h TRAIN histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    training, candidate = Path(a.training), Path(a.candidate)
    tr, tm, cr = (
        read(training / "result.json"),
        read(training / "manifest.json"),
        read(candidate / "result.json"),
    )
    plan = tm["config"]["plan"]
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    source = read(Path(plan["source"]) / "result.json")
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or cr["status"] != "completed_temporal_motion_boundary_onepass"
        or cr["program_sha256"] != sha256("scripts/train_opponent_temporal_full_30.py")
        or cr["boundary_source_sha256"]
        != sha256("src/flybrain_drive/flight_opponent_temporal_boundary.py")
        or cr["parent_boundary_sha256"]
        != sha256("src/flybrain_drive/flight_temporal_motion_boundary.py")
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or cr["dataset_sha256"] != sha256(root / "dataset.json")
        or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact original graph and trained opponent candidate required")
    entries = [e for e in dataset["episodes"] if e["role"] == "train"]
    if len(entries) != 6:
        raise ValueError("Exactly six original TRAIN histories required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Exact v7 model required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    saved = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    if (
        saved["schema"] != "flybrain.opponent_temporal_motion_boundary.v1"
        or saved["actor_qualified"] is not False
    ):
        raise ValueError("Research-only opponent checkpoint required")
    temporal = OpponentTemporalMotionBoundary(
        model.backbone.scene_command.mean,
        model.backbone.scene_command.scale,
        len(model.backbone.vision_ports),
    ).to("cuda")
    temporal.load_state_dict(saved["model"], strict=True)
    temporal.eval().requires_grad_(False)
    factors = (0.0, 1.0, 4.0, 16.0, 64.0)
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "training_checkpoint_sha256": tr["checkpoint_sha256"],
            "candidate_checkpoint_sha256": cr["checkpoint_sha256"],
            "dataset_sha256": sha256(root / "dataset.json"),
            "factors": factors,
            "selection": "minimum aggregate original TRAIN steering MSE; ties smaller gain",
            "direct_motion": "original full",
            "development_opened": False,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    sums = np.zeros((len(factors), 2), np.float64)
    counts = np.zeros(2, np.int64)
    arrays, reports = {}, []
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    for entry in entries:
        path = root / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Original TRAIN history changed")
        with load_array(path, allow_pickle=False) as data:
            times = data["observation_time"].copy()
            t = {k: torch.tensor(data[k], device="cuda") for k in names}
        complete_history(times, times)
        if t["visual_valid"].any():
            raise ValueError("Original sensory masks required")
        with torch.no_grad():
            base = model._drive_tensor(*(t[k] for k in names))
            state, traces, held, predictions = model.initial_state(len(factors)), None, None, []
            for i in range(len(times)):
                elapsed = None if i == 0 else float(times[i] - times[i - 1])
                if i:
                    for dt in substeps(elapsed):
                        state = model._advance_elapsed_tensor(held, state, dt)
                predictions.append(model._read_state_tensor(state)[0].cpu().numpy())
                delta, traces = temporal.step(
                    t["scene_features"][i : i + 1],
                    torch.ones((1, 4), dtype=torch.bool, device="cuda"),
                    traces,
                    elapsed,
                )
                motion = (traces[0] - traces[1]).reshape(1, 4, 2, 4, 16)
                opponent = torch.stack(
                    (
                        motion[:, :, :, 1] - motion[:, :, :, 0],
                        motion[:, :, :, 2] - motion[:, :, :, 3],
                    ),
                    dim=3,
                )
                pre = temporal.opponent_projection(opponent.reshape(1, 256))
                if not torch.allclose(0.5 * torch.tanh(pre), delta, rtol=0, atol=1e-7):
                    raise ValueError("Exact original temporal projection required")
                held = torch.cat(
                    [
                        temporal.modulate(
                            base[i : i + 1],
                            model.backbone.vision_ports,
                            0.5 * torch.tanh(factor * pre),
                        )
                        for factor in factors
                    ]
                )
        output = np.stack(predictions)
        with load_array(path, allow_pickle=False) as data:
            truth, valid = data["target"].copy(), data["target_valid"].copy()
        if truth.shape != (len(times), 2) or valid.shape != truth.shape or valid.dtype != bool:
            raise ValueError("Original masked labels required")
        for head in range(2):
            mask = valid[:, head]
            counts[head] += int(mask.sum())
            sums[:, head] += np.sum((output[mask, :, head] - truth[mask, head, None]) ** 2, axis=0)
        stem = entry["file"].removesuffix(".npz")
        arrays[stem + "_output"], arrays[stem + "_time"] = output, times
        reports.append(
            {"file": entry["file"], "rows": len(times), "labels": valid.sum(axis=0).tolist()}
        )
        print(json.dumps(reports[-1]), flush=True)
    if np.any(counts == 0):
        raise ValueError("Nonempty original target masks required")
    mse = sums / counts
    selected = int(np.argmin(mse[:, 0]))
    new = dict(saved)
    new["model"] = {k: v.clone() for k, v in saved["model"].items()}
    new["model"]["opponent_projection.weight"] *= factors[selected]
    new["gain_selection"] = {
        "factor": factors[selected],
        "source_candidate_sha256": cr["checkpoint_sha256"],
        "train_only": True,
        "learned_biological_value": False,
    }
    torch.save(new, run / "temporal-boundary.pt")
    np.savez_compressed(run / "predictions.npz", **arrays)
    result = {
        "status": "completed_opponent_gain_train_selection",
        "factors": factors,
        "train_mse_steering_forward": mse.tolist(),
        "label_counts": counts.tolist(),
        "selected_factor": factors[selected],
        "checkpoint_sha256": sha256(run / "temporal-boundary.pt"),
        "source_candidate_sha256": cr["checkpoint_sha256"],
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "dataset_sha256": sha256(root / "dataset.json"),
        "predictions_sha256": sha256(run / "predictions.npz"),
        "reports": reports,
        "development_opened": False,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
