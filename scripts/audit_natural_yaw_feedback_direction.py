"""Audit model feedback sign on fixed original-history input-port interventions."""

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
    parser.add_argument("--training", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training, intervention = Path(args.training), Path(args.intervention)
    manifest = read(training / "manifest.json")
    receipt = read(intervention / "result.json")
    audit_manifest = read(intervention / "manifest.json")
    predictions = intervention / "predictions.npz"
    if (receipt["status"] != "completed_natural_yaw_port_interventions"
            or sha256(predictions) != receipt["predictions_sha256"]
            or audit_manifest["config"]["training_result_sha256"]
            != sha256(training / "result.json")
            or audit_manifest["config"]["actor_deployed"] is not False):
        raise ValueError("Pinned original yaw port intervention and training required")
    root = Path(manifest["config"]["plan"]["data"])
    dataset_path = root / "dataset.json"
    dataset = read(dataset_path)
    if sha256(dataset_path) != audit_manifest["config"]["dataset_sha256"]:
        raise ValueError("Exact original dataset required")
    reports, values = [], {}
    with np.load(predictions, allow_pickle=False) as prior:
        for entry in dataset["episodes"]:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original episode hash changed")
            with load_array(path, allow_pickle=False) as data:
                body, available = data["body"].copy(), data["body_valid"].copy()
                times = data["observation_time"].copy()
            prefix = entry["file"].removesuffix(".npz")
            intact = prior[prefix + "_intact_intent"][:, 0]
            zero = prior[prefix + "_body_yaw_zero_intent"][:, 0]
            if (body.shape != (len(times), 21) or available.shape != body.shape
                    or intact.shape != times.shape or zero.shape != times.shape
                    or not all(np.isfinite(x).all() for x in (body, times, intact, zero))
                    or np.any(np.diff(times) <= 0)):
                raise ValueError("Complete ordered original histories required")
            # At decision i the circuit integrated previously held input i-1.
            rate = body[:-1, 4]
            delta = (intact-zero)[1:]
            selected = available[:-1, 4] & (np.abs(rate) > .02)
            sign = rate[selected]*delta[selected]
            subgroup = {}
            for label, side in (("positive_yaw", selected & (rate > 0)),
                                ("negative_yaw", selected & (rate < 0))):
                products = rate[side]*delta[side]
                subgroup[label] = {"samples": int(side.sum()),
                    "opposing_output_count": int((products < 0).sum()),
                    "same_direction_count": int((products > 0).sum()),
                    "zero_output_count": int((products == 0).sum())}
            reports.append({"file": entry["file"], "role": entry["role"],
                "samples": int(selected.sum()), "opposing_output_count": int((sign < 0).sum()),
                "same_direction_count": int((sign > 0).sum()),
                "zero_output_count": int((sign == 0).sum()),
                "mean_yaw_times_output_difference": float(sign.mean()) if len(sign) else None,
                "subgroups": subgroup})
            values[prefix + "_previous_yaw"] = rate
            values[prefix + "_output_difference"] = delta
            values[prefix + "_selected"] = selected
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "training_result_sha256": sha256(training / "result.json"),
        "dataset_sha256": sha256(dataset_path), "intervention_result_sha256": sha256(
            intervention / "result.json"), "predictions_sha256": sha256(predictions),
        "yaw_threshold_rad_s": .02, "steering_intent_sign": "left_positive",
        "yaw_sign": "left_positive", "actor_deployed": False})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / "signs.npz", **values)
    result = {"status": "completed_natural_yaw_feedback_direction_audit", "reports": reports,
        "signs_sha256": sha256(run / "signs.npz"), "comparison_is_full_history_intervention": True,
        "not_a_local_derivative_or_physiological_lesion": True,
        "repeated_development_not_independent": True, "physical_driving_verified": False,
        "actor_deployed": False, "optimizer_updates": 0}
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
