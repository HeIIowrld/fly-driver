"""Measure sustained and transient opponent features on saved TRAIN/DEV histories.

No labels, learned slow-path weights, driving or actuator commands are used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_sustained_opponent_boundary import (
    SustainedOpponentBoundary,
    opponent_axes,
)
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "temporal", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    training, temporal_dir = Path(args.training), Path(args.temporal)
    tr, tc = read(training / "result.json"), read(temporal_dir / "result.json")
    root = Path(read(training / "manifest.json")["config"]["plan"]["data"])
    if (sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
            or sha256(temporal_dir / "temporal-boundary.pt") != tc["checkpoint_sha256"]
            or tc["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or tc["dataset_sha256"] != sha256(root / "dataset.json")):
        raise ValueError("Original model, temporal checkpoint and dataset must match")
    saved = torch.load(temporal_dir / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    weights = saved["model"]
    boundary = SustainedOpponentBoundary(weights["mean"], weights["scale"],
                                         weights["opponent_projection.weight"].shape[0])
    missing, unexpected = boundary.load_state_dict(weights, strict=False)
    if missing != ["sustained_projection.weight"] or unexpected:
        raise ValueError("Only the new zero-initialized sustained path may be missing")
    boundary.eval().requires_grad_(False)
    entries = read(root / "dataset.json")["episodes"]
    if sum(e["role"] == "train" for e in entries) != 6 or sum(
            e["role"] == "development" for e in entries) != 2:
        raise ValueError("Original six TRAIN and two known DEV histories required")
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "boundary_sha256": sha256("src/flybrain_drive/flight_sustained_opponent_boundary.py"),
        "parent_boundary_sha256": sha256("src/flybrain_drive/flight_opponent_temporal_boundary.py"),
        "temporal_checkpoint_sha256": tc["checkpoint_sha256"],
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "dataset_sha256": tc["dataset_sha256"], "known_development_not_blind": True,
        "slow_path": "zero initialized; feature activity only", "labels_opened": False,
        "actor_deployed": False, "optimizer_updates": 0})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    reports, arrays = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original history changed")
            with load_array(path, allow_pickle=False) as data:
                times, scene = data["observation_time"].copy(), data["scene_features"].copy()
                visual_valid = data["visual_valid"].copy()
            complete_history(times, times)
            if scene.shape != (len(times), 576) or visual_valid.any():
                raise ValueError("Original four-view feature history and mask required")
            scene = torch.from_numpy(scene)
            views = torch.ones((1, 4), dtype=torch.bool)
            trace, summary = None, []
            for i in range(len(times)):
                elapsed = None if i == 0 else float(times[i] - times[i-1])
                _, trace = boundary.step(scene[i:i+1], views, trace, elapsed)
                slow = opponent_axes(trace[1]).reshape(4, 2, 2, 16)
                transient = opponent_axes(trace[0] - trace[1]).reshape(4, 2, 2, 16)
                summary.append(torch.stack((slow.square().mean(dim=(1, 2, 3)).sqrt(),
                    transient.square().mean(dim=(1, 2, 3)).sqrt())).numpy())
            summary = np.stack(summary)
            stem = entry["file"].removesuffix(".npz")
            arrays[stem + "_view_rms"] = summary
            reports.append({"file": entry["file"], "role": entry["role"],
                "observations": len(times),
                "mean_slow_rms_by_view": summary[:, 0].mean(axis=0).tolist(),
                "mean_transient_rms_by_view": summary[:, 1].mean(axis=0).tolist()})
            print(entry["file"], entry["role"], flush=True)
    np.savez_compressed(run / "view-rms.npz", **arrays)
    write_json(run / "result.json", {"status": "completed_sustained_opponent_feature_audit",
        "reports": reports, "feature_sha256": sha256(run / "view-rms.npz"),
        "labels_opened": False, "optimizer_updates": 0, "actor_deployed": False})


if __name__ == "__main__":
    main()
