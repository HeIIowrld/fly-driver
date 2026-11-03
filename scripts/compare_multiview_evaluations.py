"""Compare frozen evaluations at common time horizons, exposing early termination."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read_run(path):
    root = Path(path)
    data = json.loads((root / "evaluation.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    config = manifest["config"]["configuration"]
    rows, hashes = {}, {}
    for i, row in enumerate(data["cases"]):
        key = (row["condition"], row["case"])
        if key in rows:
            raise ValueError("Duplicate evaluation condition and case")
        file = root / f"trace-{i:02d}.npz"
        with load_array(file, allow_pickle=False) as trace:
            values = trace["trace"]
            if (values.shape != (row["steps"], 4) or not np.isfinite(values).all()
                    or not np.array_equal(trace["columns"], ["simulation_time", "steering_proxy",
                        "diagnostic_progress", "diagnostic_lateral"])
                    or np.any(np.diff(values[:, 0]) <= 0)):
                raise ValueError("Trace schema, decision count or timing differs")
        rows[key] = (row, values)
        hashes[file.name] = sha256(file)
    return config, rows, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    previous, old, old_hashes = read_run(args.previous)
    current, new, new_hashes = read_run(args.current)
    for key in ("control_dt_seconds", "fixed_speed_mps", "brain_ticks_per_decision",
                "period_decisions", "view_maps_sha256"):
        if previous[key] != current[key]:
            raise ValueError("Matched environment and timing contracts required: " + key)
    if previous["evaluation"] != current["evaluation"]:
        raise ValueError("Same evaluation cases, conditions and horizon required")
    run = begin_run(args.output, {"arguments": vars(args),
        "previous_trace_hashes": old_hashes, "current_trace_hashes": new_hashes,
        "scope": "development fixed-speed MuJoCo comparison, not independent AC validation"})
    conditions = ("normal", "vision_missing")
    case_names = [case["name"] for case in current["evaluation"]["cases"]]
    budget = current["evaluation"]["steps"]
    pooled = {k: [] for k in ("previous_normal", "previous_vision_missing",
                              "current_normal", "current_vision_missing")}
    results = []
    for name in case_names:
        horizon = min(len(source[(condition, name)][1]) for source in (old, new)
                      for condition in conditions)
        row = {"case": name, "matched_decisions": horizon,
               "matched_seconds": horizon * current["control_dt_seconds"]}
        for label, source in (("previous", old), ("current", new)):
            for condition in conditions:
                summary, values = source[(condition, name)]
                errors = values[:horizon, 3]
                key = label + "_" + condition
                pooled[key].append(errors)
                row[key] = {"common_horizon_lateral_rms_m": float(np.sqrt(np.mean(errors**2))),
                    "recorded_decisions": len(values), "termination": summary["reason"],
                    "full_budget_without_exit": len(values) == budget
                        and summary["reason"] in ("time_limit", "evaluation_budget"),
                    "progress_delta_m": summary["progress_delta_m"]}
        results.append(row)
    rms = {key: float(np.sqrt(np.mean(np.concatenate(values)**2)))
           for key, values in pooled.items()}
    write_json(run / "comparison.json", {"cases": results,
        "pooled_common_horizon_lateral_rms_m": rms,
        "current_visual_error_lower_than_missing":
            rms["current_normal"] < rms["current_vision_missing"],
        "current_error_lower_than_previous": rms["current_normal"] < rms["previous_normal"],
        "complete_lap_verified": False, "multi_view_efficiency_advantage_verified": False,
        "AC_ready": False, "goal_achieved": False})
    print(json.dumps(rms, indent=2))


if __name__ == "__main__":
    main()
