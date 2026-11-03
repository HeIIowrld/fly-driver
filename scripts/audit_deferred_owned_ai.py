"""Compare source-frame render timing in two bounded owned AI motion diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def audit(root, deferred):
    result = read(root / "result.json")
    rows_path = root / "diagnostic-frames.json"
    receipt = read(root / "control/api-receipt.json")
    control = read(root / "control/result.json")
    if (
        result["status"] != "completed"
        or sha256(rows_path) != result["metadata_sha256"]
        or control["status"] != "completed"
        or receipt["status"] != "api_returned_true"
        or not receipt["motion_probe_passed"]
    ):
        raise ValueError("Completed owned original motion diagnostics required")
    rows = read(rows_path)
    spans, velocities, moving_indices = [], [], []
    for row in rows:
        path = Path(row["source_file"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Original diagnostic frame changed")
        frame = read(path)
        diag = frame["airflow_diagnostic"]
        if (
            frame["request_id"] != result["request_id"]
            or diag != row["diagnostic"]
            or not frame["aid_probe_only"]
            or diag["actor_admissible"] is not False
            or diag["source_verified"] is not False
            or bool(frame.get("deferred_save_probe", False)) != deferred
        ):
            raise ValueError("Original diagnostic identity and isolation required")
        views = frame["views"]
        if len(views) != 4 or {v["view_id"] for v in views} != {"front", "left", "right", "rear"}:
            raise ValueError("All four original views required")
        finishes = [v["render_call_finished_clock"] for v in views]
        if deferred and any(v["save_finished_clock"] < max(finishes) for v in views):
            raise ValueError("Deferred saves must follow all four renders")
        speed = float(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]))
        if speed > 1:
            starts = [v["render_started_clock"] for v in views]
            spans.append((max(starts) - min(starts)) * 1000)
            velocities.append(speed * 3.6)
            moving_indices.append(row["index"])
    if len(spans) < 10:
        raise ValueError("Sufficient original moving diagnostic observations required")
    return {
        "frames": len(rows),
        "moving_frames_speed_over_1m_s": len(spans),
        "moving_frame_indices": moving_indices,
        "moving_render_span_ms": spans,
        "moving_span_median_ms": float(np.median(spans)),
        "moving_span_p95_ms": float(np.percentile(spans, 95)),
        "moving_span_max_ms": max(spans),
        "moving_speed_max_kmh": max(velocities),
        "source_frames_reverified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = [Path(args.baseline), Path(args.candidate)]
    paths = [
        r / p
        for r in roots
        for p in (
            "result.json",
            "diagnostic-frames.json",
            "control/api-receipt.json",
            "control/result.json",
        )
    ]
    pins = {str(p): sha256(p) for p in paths}
    before, after = audit(roots[0], False), audit(roots[1], True)
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original evidence changed during audit")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "different bounded owned AI runs, moving native speed candidate above 1m/s",
            "physical_exposure_synchronization_not_verified": True,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_owned_ai_deferred_timing_comparison",
        "baseline": before,
        "deferred": after,
        "independent_runs_not_same_physics_trajectory": True,
        "physical_exposure_sync_verified": False,
        "flight_actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
