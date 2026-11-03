"""Counterfactual clipping audit of a fixed failed visual readout; no model selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_joint_view_expanded_histories import prepare, read


def rmse(prediction, target, valid):
    return float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))


def region(mask, valid, prediction, target):
    selected = mask & valid
    return {
        "valid_rows": int(selected.sum()),
        "original_rmse_per_m": rmse(prediction, target, selected) if selected.any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old-dataset", "old-sensors", "new-dataset", "new-sensors", "ridge", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root = Path(args.ridge)
    receipt, manifest = read(root / "result.json"), read(root / "manifest.json")["config"]
    checkpoint = root / "joint-view-ridge.npz"
    if (
        receipt["status"] != "completed_train_group_crossvalidated_joint_ridge"
        or sha256(checkpoint) != receipt["checkpoint_sha256"]
        or sha256(args.old_dataset) != manifest["older_dataset_sha256"]
        or sha256(args.old_sensors) != manifest["older_readout_sha256"]
    ):
        raise ValueError("Exact frozen model and older original sources required")
    entries = read(args.old_dataset)["episodes"] + read(args.new_dataset)["episodes"]
    all_sensors = read(args.old_sensors)["episodes"] + read(args.new_sensors)["episodes"]
    sources = {x["id"]: x for x in all_sensors}
    if (
        len(entries) != 14
        or len(sources) != 14
        or len({e["id"] for e in entries}) != 14
        or any(e["role"] != sources[e["id"]]["role"] for e in entries)
    ):
        raise ValueError("All original role-bound histories required")
    with np.load(checkpoint, allow_pickle=False) as f:
        mean, scale, weights = (
            f["feature_mean"].copy(),
            f["feature_scale"].copy(),
            f["weights"].copy(),
        )
        bias = float(f["target_mean"])
    prior = {e["id"]: e for e in receipt["development_reports"]}
    reports, all_pins = [], {}
    for entry in entries:
        if entry["role"] != "development":
            continue
        r = prepare(entry, sources[entry["id"]])
        all_pins.update(r["pins"])
        z = (r["x"] - mean) / scale
        clipped = np.clip(z, -5, 5)
        original = bias + clipped @ weights
        raw = bias + z @ weights
        valid = r["valid"]
        y = r["y"]
        if abs(rmse(original, y, valid) - prior[entry["id"]]["rmse_per_m"]) > 1e-10:
            raise ValueError("Original fixed prediction differs")
        excess = z - clipped
        effects = excess * weights[None, :]
        group_effects = []
        for view_index, view in enumerate(("front", "left", "right", "rear")):
            for family, offset, count in (("retinal", 0, 6), ("motion", 384, 8)):
                start, end = (
                    offset + view_index * count * 16,
                    offset + (view_index + 1) * count * 16,
                )
                group_effects.append(
                    {
                        "view": view,
                        "family": family,
                        "clipped_values": int((np.abs(z[:, start:end]) > 5).sum()),
                        "mean_abs_unclipping_contribution": float(
                            np.abs(effects[valid, start:end].sum(axis=1)).mean()
                        ),
                    }
                )
        row_hit = np.any(np.abs(z) > 5, axis=1)

        reports.append(
            {
                "id": entry["id"],
                "valid_rows": int(valid.sum()),
                "original_rmse_per_m": rmse(original, y, valid),
                "unclipped_rmse_per_m": rmse(raw, y, valid),
                "mean_abs_prediction_difference": float(
                    np.abs(raw[valid] - original[valid]).mean()
                ),
                "max_abs_prediction_difference": float(np.abs(raw[valid] - original[valid]).max()),
                "valid_rows_with_clipping": region(row_hit, valid, original, y),
                "valid_rows_without_clipping": region(~row_hit, valid, original, y),
                "view_family_effects": group_effects,
            }
        )
        print(json.dumps({"audited": entry["id"]}), flush=True)
    inputs = [
        args.old_dataset,
        args.old_sensors,
        args.new_dataset,
        args.new_sensors,
        str(checkpoint),
        str(root / "result.json"),
    ]
    all_pins.update({path: sha256(path) for path in inputs})
    if any(sha256(path) != digest for path, digest in all_pins.items()):
        raise ValueError("Original evidence changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": all_pins,
            "scope": "post hoc fixed-readout counterfactual; not deployment or new validation",
            "view_order": ["front", "left", "right", "rear"],
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_four_view_clipping_effect_audit",
        "reports": reports,
        "known_dev_is_not_independent": True,
        "unclipped_not_qualified": True,
        "no_images_or_biological_neurons_ablated": True,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
