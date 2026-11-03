"""Compare visual-motion interventions with previously held body-yaw sign."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "motion", "opposite", "output"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    training, motion, opposite = map(Path, (a.training, a.motion, a.opposite))
    tr, mm, om = (
        read(training / "result.json"),
        read(motion / "manifest.json"),
        read(opposite / "manifest.json"),
    )
    mr, ore = read(motion / "result.json"), read(opposite / "result.json")
    if (
        mr["status"] != "completed_dnb06_original_history_state_audit"
        or ore["status"] != "completed_centered_opposite_motion_channel_replay"
        or sha256(motion / "predictions.npz") != mr["predictions_sha256"]
        or sha256(opposite / "predictions.npz") != ore["predictions_sha256"]
        or om["config"]["prior_predictions_sha256"] != mr["predictions_sha256"]
        or om["config"]["training_result_sha256"] != sha256(training / "result.json")
        or om["config"]["candidate_checkpoint_sha256"]
        != mm["config"]["candidate_checkpoint_sha256"]
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
    ):
        raise ValueError("Exact paired visual intervention evidence required")
    root = Path(read(training / "manifest.json")["config"]["plan"]["data"])
    if (
        sha256(root / "dataset.json") != mm["config"]["dataset_sha256"]
        or sha256(root / "dataset.json") != om["config"]["dataset_sha256"]
    ):
        raise ValueError("Exact source dataset required")
    reports, values = [], {}
    with (
        load_array(motion / "predictions.npz", allow_pickle=False) as original,
        load_array(opposite / "predictions.npz", allow_pickle=False) as swapped,
    ):
        for entry in read(root / "dataset.json")["episodes"]:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original body history changed")
            with load_array(path, allow_pickle=False) as data:
                body, mask, times = (
                    data["body"].copy(),
                    data["body_valid"].copy(),
                    data["observation_time"].copy(),
                )
            prefix = entry["file"].removesuffix(".npz")
            intact = original[prefix + "_intact_intent"][:, 0]
            mean = original[prefix + "_motion_mean_intent"][:, 0]
            swapped_intact = swapped[prefix + "_intact_intent"][:, 0]
            other = swapped[prefix + "_opposite_motion_channels_intent"][:, 0]
            np.testing.assert_allclose(intact, swapped_intact, rtol=0, atol=1e-6)
            if (
                body.shape != (len(times), 21)
                or mask.shape != body.shape
                or intact.shape != times.shape
                or not all(np.isfinite(v).all() for v in (body, times, intact, mean, other))
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError("Ordered finite original body and visual histories required")
            yaw = body[:-1, 4]
            valid = mask[:-1, 4] & (np.abs(yaw) > 0.02)
            report = {
                "file": entry["file"],
                "role": entry["role"],
                "samples": int(valid.sum()),
                "conditions": {},
            }
            for name, delta in (
                ("motion_vs_mean", (intact - mean)[1:]),
                ("motion_vs_centered_opposite", (intact - other)[1:]),
            ):
                product = yaw[valid] * delta[valid]
                report["conditions"][name] = {
                    "opposes_previous_yaw": int((product < 0).sum()),
                    "same_direction_as_previous_yaw": int((product > 0).sum()),
                    "zero": int((product == 0).sum()),
                    "mean_abs_steering_effect": float(np.abs(delta[valid]).mean())
                    if valid.any()
                    else None,
                }
                values[prefix + "_" + name] = delta
            values[prefix + "_previous_yaw"], values[prefix + "_selected"] = yaw, valid
            reports.append(report)
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "dataset_sha256": sha256(root / "dataset.json"),
            "motion_sha256": mr["predictions_sha256"],
            "opposite_sha256": ore["predictions_sha256"],
            "known_development_not_blind": True,
            "yaw_threshold_rad_s": 0.02,
            "sign_contract": "left-positive yaw and steering",
            "scope": "full-history visual feature intervention compared to previous body yaw",
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / "signs.npz", **values)
    write_json(
        run / "result.json",
        {
            "status": "completed_visual_motion_yaw_sign_audit",
            "reports": reports,
            "signs_sha256": sha256(run / "signs.npz"),
            "not_local_derivative": True,
            "not_physiology_or_closed_loop_validation": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
