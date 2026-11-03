"""Fit a diagnostic 32-cell motion-to-body-yaw probe with capture-group holdouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_joint_view_expanded_histories import read


def group(name):
    if name.startswith("original-"):
        return "original"
    return "spatial" if name.startswith("spatial-") else "exterior"


def fit(records, penalty):
    x = np.concatenate([r["x"] for r in records])
    y = np.concatenate([r["y"] for r in records])
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    z = np.clip((x - mean) / scale, -5, 5)
    intercept = float(y.mean())
    matrix = z.T @ z + np.eye(z.shape[1]) * len(z) * penalty
    weights = np.linalg.solve(matrix, z.T @ (y - intercept))
    return mean, scale, intercept, weights


def report(model, record):
    mean, scale, intercept, weights = model
    y = record["y"]
    predicted = intercept + np.clip((record["x"] - mean) / scale, -5, 5) @ weights
    return {
        "id": record["id"],
        "rows": len(y),
        "rmse_rad_s": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "train_constant_rmse_rad_s": float(np.sqrt(np.mean((intercept - y) ** 2))),
        "zero_rmse_rad_s": float(np.sqrt(np.mean(y * y))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-sensors", required=True)
    parser.add_argument("--new-sensors", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    entries = read(args.old_sensors)["episodes"] + read(args.new_sensors)["episodes"]
    if len(entries) != 14 or len({e["id"] for e in entries}) != 14:
        raise ValueError("Original 14 distinct histories required")
    pins = {p: sha256(p) for p in (args.old_sensors, args.new_sensors)}
    records = []
    # The held-out DEV arrays are not loaded until after training and checkpoint publication.
    for e in entries:
        if e["role"] != "train":
            continue
        records.append(load_record(e, pins))
        print(json.dumps({"loaded_train": e["id"]}), flush=True)
    if len(records) != 11 or sum(e["role"] == "development" for e in entries) != 3:
        raise ValueError("Original role counts required")
    penalties = [0.1, 1.0, 10.0]
    folds = []
    for held in ("original", "spatial", "exterior"):
        train, test = (
            [r for r in records if group(r["id"]) != held],
            [r for r in records if group(r["id"]) == held],
        )
        if not train or not test:
            raise ValueError("Each capture group must exist")
        for penalty in penalties:
            model = fit(train, penalty)
            folds.append(
                {
                    "held_group": held,
                    "penalty": penalty,
                    "fit_ids": [r["id"] for r in train],
                    "reports": [report(model, r) for r in test],
                }
            )
    losses = {
        str(p): float(
            np.mean([r["rmse_rad_s"] for f in folds if f["penalty"] == p for r in f["reports"]])
        )
        for p in penalties
    }
    selected = min(penalties, key=lambda p: (losses[str(p)], p))
    model = fit(records, selected)
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins.copy(),
            "input": "four views x eight original motion cells, mean of original 721 receptors",
            "target": "same decision yaw body sensor; never added as actor input",
            "selection": "mean TRAIN capture-group held-out episode RMSE",
            "penalties": penalties,
            "development_opened_before_checkpoint": False,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    mean, scale, intercept, weights = model
    np.savez_compressed(
        run / "yaw-probe.npz",
        feature_mean=mean,
        feature_scale=scale,
        intercept=intercept,
        weights=weights,
        penalty=selected,
    )
    checkpoint_hash = sha256(run / "yaw-probe.npz")
    dev = [load_record(e, pins) for e in entries if e["role"] == "development"]
    metrics = [report(model, r) for r in dev]
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original input changed")
    out = {
        "status": "completed_four_view_yaw_diagnostic_fit",
        "folds": folds,
        "train_fold_mean_rmse": losses,
        "selected_penalty": selected,
        "development_reports": metrics,
        "checkpoint_sha256": checkpoint_hash,
        "source_hashes_including_development": pins,
        "original_visual_weights_changed": False,
        "development_opened_before_checkpoint": False,
        "known_dev_not_independent_final_test": True,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps({"selected_penalty": selected, "development_reports": metrics}), flush=True)


def load_record(e, pins):
    visual = Path(e["output"]) / "responses.npz"
    body_dir = Path(e["body_replay"])
    body, body_result = body_dir / "virtual-body.npz", body_dir / "result.json"
    receipt = read(body_result)
    if (
        sha256(visual) != e["responses_sha256"]
        or receipt["status"] != "completed"
        or sha256(body) != receipt["virtual_body_sha256"]
    ):
        raise ValueError("Original response and body hashes required")
    pins.update({str(p): sha256(p) for p in (visual, body, body_result)})
    with load_array(visual, allow_pickle=False) as data:
        x = np.asarray(data["motion_activity"], dtype=np.float64).mean(axis=3).reshape(-1, 32)
        clock = data["decision_monotonic_seconds"].copy()
    with load_array(body, allow_pickle=False) as data:
        if (
            not np.array_equal(clock, data["decision_monotonic_seconds"])
            or data["feature_names"][4] != "angular_body.yaw_left_rad_s"
        ):
            raise ValueError("Original matched decision clocks and yaw convention required")
        y, valid = data["values"][:, 4].copy(), data["valid"][:, 4].copy()
    if (
        x.shape != (e["observations"], 32)
        or valid.sum() < 50
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Finite original responses and sufficient valid body yaw required")
    return {"id": e["id"], "x": x[valid], "y": y[valid]}


if __name__ == "__main__":
    main()
