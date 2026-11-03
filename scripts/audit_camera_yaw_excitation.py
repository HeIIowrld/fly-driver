"""Measure turning excitation in retained camera diagnostics, never actor inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def inspect(root, *, max_interval_seconds):
    root = Path(root)
    receipt = read(root / "result.json")
    rows_path = root / "diagnostic-frames.json"
    if receipt["status"] != "completed" or sha256(rows_path) != receipt["metadata_sha256"]:
        raise ValueError("Completed recording and original metadata hash required")
    rows = read(rows_path)
    if len(rows) != receipt["frames"] or len(rows) < 2:
        raise ValueError("Complete nonempty history required")
    times, forward, speed = [], [], []
    for i, row in enumerate(rows):
        path = Path(row["source_file"])
        if row["index"] != i or sha256(path) != row["sha256"]:
            raise ValueError("Original frame order and hash required")
        frame = read(path)
        diag = frame["airflow_diagnostic"]
        if diag != row["diagnostic"] or diag["actor_admissible"] is not False:
            raise ValueError("Exact diagnostic-only source required")
        times.append(diag["started_source_clock"])
        forward.append(diag["body_forward_candidate"])
        speed.append(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]))
    times, forward, speed = map(np.asarray, (times, forward, speed))
    if (
        forward.shape != (len(rows), 3)
        or not all(np.isfinite(x).all() for x in (times, forward, speed))
        or np.any(np.diff(times) <= 0)
        or np.any(np.linalg.norm(forward[:, [0, 2]], axis=1) < 0.9)
    ):
        raise ValueError("Finite, ordered horizontal forward direction required")
    # World azimuth is diagnostic truth only; it is never an actor observation.
    heading = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    dt = np.diff(times)
    rate = np.diff(heading) / dt
    eligible = (speed[:-1] > 1) & (speed[1:] > 1) & (dt <= max_interval_seconds)
    selected = rate[eligible]
    return {
        "recording": str(root),
        "recording_result_sha256": sha256(root / "result.json"),
        "metadata_sha256": sha256(rows_path),
        "frames": len(rows),
        "moving_intervals": len(selected),
        "horizontal_heading_span_rad": float(np.ptp(heading)),
        "moving_absolute_heading_rate_p95_rad_s": (
            float(np.quantile(np.abs(selected), 0.95)) if len(selected) else None
        ),
        "moving_intervals_rate_above_positive_0_02": int((selected > 0.02).sum()),
        "moving_intervals_rate_below_negative_0_02": int((selected < -0.02).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-interval-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if not np.isfinite(args.max_interval_seconds) or not 0 < args.max_interval_seconds <= 0.25:
        raise ValueError("Finite interval bound no longer than 0.25 seconds required")
    reports = [inspect(root, max_interval_seconds=args.max_interval_seconds)
               for root in args.recording]
    run = begin_run(args.output, {
        "program_sha256": sha256(__file__),
        "scope": "diagnostic excitation only; world azimuth excluded from actor",
        "speed_cutoff_m_s": 1.0,
        "max_interval_seconds": args.max_interval_seconds,
        "rate_threshold_rad_s": 0.02,
        "threshold_is_engineering_screen_not_physiological_validation": True,
        "actor_deployed": False,
    })
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = {"status": "completed_yaw_excitation_audit", "recordings": reports,
              "source_time_or_physical_exposure_sync_verified": False,
              "axis_sign_verified": False, "actor_deployed": False}
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
