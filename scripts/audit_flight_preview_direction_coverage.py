"""Audit two-sided teacher-path coverage against fixed flight steering predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "labels", "evaluation", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    dataset, labels, evaluation = map(Path, (a.dataset, a.labels, a.evaluation))
    data = read(dataset / "dataset.json")
    lr, lm = read(labels / "result.json"), read(labels / "manifest.json")["config"]
    er, em = read(evaluation / "result.json"), read(evaluation / "manifest.json")["config"]
    if (
        lr["status"] != "completed_visual_preview_information_probe"
        or er["status"] != "completed_hsvs_isolated_motion_comparison"
        or lm["parent_sha256"] != sha256(dataset / "dataset.json")
        or sha256(evaluation / "predictions.npz") != er["predictions_sha256"]
        or em["actor_deployed"] is not False
    ):
        raise ValueError("Exact original dataset, teacher labels and fixed evaluation required")
    entries = {e["file"]: e for e in data["episodes"]}
    label_entries = {e["file"]: e for e in lr["episodes"]}
    if entries.keys() != label_entries.keys():
        raise ValueError("Exact original episode identities required")
    episodes, training_counts = [], np.zeros((3, 2), dtype=int)
    threshold = 0.001  # Explicit diagnostic screen in 1/m, not a biological threshold.
    with load_array(evaluation / "predictions.npz", allow_pickle=False) as predictions:
        for name, entry in entries.items():
            le = label_entries[name]
            if (
                entry["role"] != le["role"]
                or entry["sha256"] != le["original_episode_sha256"]
                or sha256(dataset / name) != entry["sha256"]
                or sha256(labels / name) != le["auxiliary_sha256"]
            ):
                raise ValueError("Role-bound original episode or target changed")
            with (
                load_array(dataset / name, allow_pickle=False) as x,
                load_array(labels / name, allow_pickle=False) as y,
            ):
                clock, label_clock = x["observation_time"].copy(), y["observation_time"].copy()
                preview, valid = y["preview"].copy(), y["preview_valid"].copy()
            np.testing.assert_array_equal(clock, label_clock)
            stem = name.removesuffix(".npz")
            output = {
                model: predictions[stem + "_" + model][:, 0].copy()
                for model in ("original", "candidate")
            }
            if (
                preview.shape != (len(clock), 3, 2)
                or valid.shape != (len(clock), 3)
                or valid.dtype != bool
                or any(v.shape != clock.shape for v in output.values())
                or not np.isfinite(preview).all()
                or not all(np.isfinite(v).all() for v in output.values())
            ):
                raise ValueError("Finite aligned target and fixed predictions required")
            curvature = preview[:, :, 1]
            if entry["role"] == "train":
                training_counts[:, 0] += ((curvature < -threshold) & valid).sum(axis=0)
                training_counts[:, 1] += ((curvature > threshold) & valid).sum(axis=0)
            episodes.append((name, entry["role"], curvature, valid, output))
    if (training_counts.sum(axis=1) == 0).any():
        raise ValueError("Nonempty training targets needed for simple majority baseline")
    # Screen-right positive target and left-positive model steering have opposite signs.
    majority = np.where(training_counts[:, 1] >= training_counts[:, 0], 1, -1)
    reports = []
    for name, role, curvature, valid, output in episodes:
        horizons = []
        for h, distance in enumerate((5, 10, 20)):
            active = valid[:, h] & (np.abs(curvature[:, h]) > threshold)
            left = active & (curvature[:, h] < 0)
            right = active & (curvature[:, h] > 0)
            outcomes = {}
            for model, steering in output.items():
                matches = np.sign(steering[active]) == -np.sign(curvature[active, h])
                outcomes[model] = {"correct": int(matches.sum()), "count": int(active.sum())}
            outcomes["training_majority_constant"] = {
                "correct": int((np.sign(curvature[active, h]) == majority[h]).sum()),
                "count": int(active.sum()),
            }
            horizons.append(
                {
                    "distance_m": distance,
                    "positive_count": int(right.sum()),
                    "negative_count": int(left.sum()),
                    "two_sided": bool(left.any() and right.any()),
                    "outcomes": outcomes,
                }
            )
        prediction_signs = {
            model: {"negative": int((steering < 0).sum()),
                    "positive": int((steering > 0).sum()),
                    "zero": int((steering == 0).sum())}
            for model, steering in output.items()
        }
        reports.append({"file": name, "role": role, "horizons": horizons,
                        "all_observation_steering_signs": prediction_signs})
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "dataset_sha256": sha256(dataset / "dataset.json"),
            "label_result_sha256": sha256(labels / "result.json"),
            "predictions_sha256": er["predictions_sha256"],
            "threshold_1_m": threshold,
            "teacher_target_scope": "future path diagnostic only, never actor input",
            "baseline": "majority sign of TRAIN valid targets per distance",
            "known_development_not_independent": True,
            "optimizer_updates": 0,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        run / "result.json",
        {
            "status": "completed_preview_direction_coverage_diagnostic",
            "train_negative_positive_counts": training_counts.tolist(),
            "reports": reports,
            "curve_sign_not_steering_physics_or_optimal_line": True,
            "independent_bilateral_validation": False,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )
    print(
        json.dumps(
            {
                "training_counts": training_counts.tolist(),
                "development": [r for r in reports if r["role"] == "development"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
