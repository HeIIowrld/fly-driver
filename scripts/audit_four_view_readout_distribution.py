"""Audit frozen four-view preview input support, without fitting or actor use."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_joint_view_expanded_histories import prepare, read, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old-dataset", "old-sensors", "new-dataset", "new-sensors", "ridge", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root = Path(args.ridge)
    checkpoint = root / "joint-view-ridge.npz"
    receipt, manifest = read(root / "result.json"), read(root / "manifest.json")["config"]
    if (
        receipt["status"] != "completed_train_group_crossvalidated_joint_ridge"
        or sha256(checkpoint) != receipt["checkpoint_sha256"]
        or sha256(args.old_dataset) != manifest["older_dataset_sha256"]
        or sha256(args.old_sensors) != manifest["older_readout_sha256"]
    ):
        raise ValueError("Exact frozen ridge and original older sources required")
    inputs = [
        args.old_dataset,
        args.old_sensors,
        args.new_dataset,
        args.new_sensors,
        str(checkpoint),
        str(root / "result.json"),
    ]
    pins = {path: sha256(path) for path in inputs}
    entries = read(args.old_dataset)["episodes"] + read(args.new_dataset)["episodes"]
    sensors = read(args.old_sensors)["episodes"] + read(args.new_sensors)["episodes"]
    sources = {item["id"]: item for item in sensors}
    if (
        len(entries) != 14
        or len(sources) != 14
        or len({x["id"] for x in entries}) != 14
        or any(x["role"] != sources[x["id"]]["role"] for x in entries)
    ):
        raise ValueError("Original unique role-bound episodes required")
    with np.load(checkpoint, allow_pickle=False) as data:
        mean, scale = data["feature_mean"].copy(), data["feature_scale"].copy()
        weights, target_mean = data["weights"].copy(), float(data["target_mean"])
    records = []
    for entry in entries:
        record = prepare(entry, sources[entry["id"]])
        records.append(record)
        print(json.dumps({"loaded": entry["id"]}), flush=True)
    train = [record for record in records if record["role"] == "train"]
    joined = np.concatenate([r["x"] for r in train])
    error_mean = float(np.max(np.abs(joined.mean(axis=0) - mean)))
    error_scale = float(np.max(np.abs(np.maximum(joined.std(axis=0), 1e-5) - scale)))
    if max(error_mean, error_scale) > 1e-10:
        raise ValueError("TRAIN histories differ from stored normalization")
    target_values = np.concatenate([r["y"][r["valid"]] for r in train])
    y_min, y_max = float(target_values.min()), float(target_values.max())
    report = []
    previous = {r["id"]: r for r in receipt["development_reports"]}
    for record in records:
        z = (record["x"] - mean) / scale
        pred = target_mean + np.clip(z, -5, 5) @ weights
        y = record["y"][record["valid"]]
        if not len(y):
            raise ValueError("Valid target support required")
        report.append(
            {
                "id": record["id"],
                "role": record["role"],
                "valid_rows": len(y),
                "clipped_feature_fraction": float((np.abs(z) > 5).mean()),
                "rows_with_clipped_feature_fraction": float(np.any(np.abs(z) > 5, axis=1).mean()),
                "median_mean_abs_standardized_feature": float(np.median(np.abs(z).mean(axis=1))),
                "target_range_per_m": [float(y.min()), float(y.max())],
                "target_outside_global_train_range_fraction": float(
                    ((y < y_min) | (y > y_max)).mean()
                ),
                "rmse_per_m": score(record, pred, target_mean)["rmse_per_m"],
            }
        )
        if record["role"] == "development":
            prior = previous[record["id"]]["rmse_per_m"]
            if abs(report[-1]["rmse_per_m"] - prior) > 1e-10:
                raise ValueError("Fixed known-development prediction differs")
    for record in records:
        if any(sha256(path) != digest for path, digest in record["pins"].items()):
            raise ValueError("Original response or target changed")
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Pinned source changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "retrospective distribution audit; no validation or selection",
            "feature_clipping_threshold": 5,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_frozen_four_view_distribution_audit",
        "reports": report,
        "train_target_range_per_m": [y_min, y_max],
        "train_mean_max_abs_error": error_mean,
        "train_scale_max_abs_error": error_scale,
        "known_dev_is_not_independent": True,
        "clipping_does_not_prove_generalization_cause": True,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
