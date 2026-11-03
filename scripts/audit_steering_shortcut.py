"""Quantify teacher-label copying from measured steering encoded in actor organ features."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from flybrain_drive.evidence import begin_run, write_json
    from flybrain_drive.graph import sha256

    source = Path(args.dataset)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    config = manifest["config"]["policy_configuration"]
    vehicle = config["environment"]["vehicle"]
    specs = config["sensory_preprocessing"]["position_organs"]
    offset = 0
    for spec in specs:
        if spec["sensor"] == "steering_angle":
            break
        offset += spec["channels"] * (spec["populations"] + 2)
    else:
        raise ValueError("Steering organ must be declared")
    if spec["channels"] != 2:
        raise ValueError("Ackermann diagnostic requires both steering angles")
    population = spec["populations"]
    centers = np.linspace(spec["minimum"], spec["maximum"], population)
    pair = max(0, population // 2 - 1)
    paths = sorted(source.glob("episode-*.npz"))
    run = begin_run(args.output, {"dataset_manifest_sha256": sha256(source / "manifest.json"),
                                 "dataset_files_sha256": {p.name: sha256(p) for p in paths},
                                 "script_sha256": sha256(__file__),
                                 "scope": "offline measured-steering copy diagnostic, not driving"})
    labels, copies, validities, early = [], [], [], []
    for path in paths:
        with load_array(path, allow_pickle=False) as stored:
            organs, mask = stored["obs_organs"], stored["obs_organ_valid"]
            angles, validity = [], []
            for side in range(2):
                j = offset + side * (population + 2) + pair
                ratio = np.maximum(organs[:, j], 1e-20) / np.maximum(organs[:, j + 1], 1e-20)
                theta = (centers[pair] + centers[pair + 1]) / 2 - (
                    spec["tuning_width"]**2 / (centers[pair + 1] - centers[pair])) * np.log(ratio)
                angles.append(theta)
                validity.append((mask[:, j] > 0) & (mask[:, j + 1] > 0))
            tangent_left, tangent_right = map(np.tan, angles)
            wheelbase, track = vehicle.get("wheelbase", 1.6), vehicle.get("track", 1.1)
            k_left = tangent_left / (wheelbase + track / 2 * tangent_left)
            k_right = tangent_right / (wheelbase - track / 2 * tangent_right)
            copies.extend(np.clip((k_left + k_right)
                                   / (2 * vehicle.get("max_curvature", .3)), -1, 1))
            labels.extend(stored["action"][:, 1])
            validities.extend(validity[0] & validity[1])
            early.extend(np.arange(len(organs)) < 50)
    labels, copies = np.array(labels), np.array(copies)
    valid, early = np.array(validities), np.array(early)
    result = {}
    for name, keep in (("all", valid), ("first_50_decisions", valid & early),
                       ("later", valid & ~early),
                       ("small_teacher_action", valid & (abs(labels) <= .1)),
                       ("large_teacher_action", valid & (abs(labels) > .1))):
        error = copies[keep] - labels[keep]
        result[name] = {"samples": int(keep.sum()),
                        "mse": float(np.mean(error**2)) if len(error) else None,
                        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None}
    write_json(run / "summary.json", {"label_copy": result,
        "valid_fraction": float(valid.mean()), "true_training_epochs": 0,
        "interpretation": "Low copying loss alone does not prove visual use or corrective driving"})
    write_json(run / "result.json", {"status": "completed", "goal_achieved": False})


if __name__ == "__main__":
    main()
