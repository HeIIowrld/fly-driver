"""Verify per-view graphics frame and car-state timestamp in an owned AI diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result, receipt = read(root / "result.json"), read(root / "control/api-receipt.json")
    rows_path, host_path = root / "diagnostic-frames.json", root / "live-frame-hashes.jsonl"
    if (
        result["status"] != "completed"
        or not result["host_hashes_observed_after_publication_not_exposure"]
        or sha256(rows_path) != result["metadata_sha256"]
        or sha256(host_path) != result["host_observed_hashes_sha256"]
        or receipt["status"] != "api_returned_true"
        or not receipt["motion_probe_passed"]
    ):
        raise ValueError("Completed owned probe and original image hash log required")
    rows = read(rows_path)
    hosts = [json.loads(line) for line in host_path.read_text(encoding="utf8").splitlines()]
    if len(rows) != len(hosts) or len(rows) != result["frames"]:
        raise ValueError("Exact complete diagnostic and host hash counts required")
    same_frame, same_timestamp, same_post = 0, 0, 0
    moving_span, all_span, timestamps, hash_duration = [], [], [], []
    for i, (row, host) in enumerate(zip(rows, hosts, strict=True)):
        path = Path(row["source_file"])
        if row["index"] != i or host["index"] != i or sha256(path) != row["sha256"]:
            raise ValueError("Original frame ordering and hashes required")
        frame = read(path)
        diag = frame["airflow_diagnostic"]
        if (
            diag != row["diagnostic"]
            or host["metadata_sha256"] != row["sha256"]
            or not frame["aid_probe_only"]
            or not frame["deferred_save_probe"]
            or diag["actor_admissible"] is not False
            or diag["source_verified"] is not False
        ):
            raise ValueError("Diagnostic identity and actor isolation required")
        views = frame["views"]
        if len(views) != 4 or {v["view_id"] for v in views} != {"front", "left", "right", "rear"}:
            raise ValueError("Four exact source views required")
        values = np.array(
            [
                v[k]
                for v in views
                for k in ("diagnostic_car_timestamp_before_ms", "diagnostic_car_timestamp_after_ms")
            ]
        )
        graphics = {
            v[k]
            for v in views
            for k in ("diagnostic_graphics_frame_before", "diagnostic_graphics_frame_after")
        }
        if not np.isfinite(values).all():
            raise ValueError("Finite candidate physics-record timestamps required")
        same_frame += int(graphics == {str(frame["render_frame"])})
        same_timestamp += int(np.all(values == values[0]))
        same_post += int(np.all(values == diag["car_physics_timestamp_ms"]))
        timestamps.append(float(values[0]))
        span = (
            max(v["render_started_clock"] for v in views)
            - min(v["render_started_clock"] for v in views)
        ) * 1000
        all_span.append(span)
        if np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]) > 1:
            moving_span.append(span)
        hash_duration.append((host["host_hash_finished"] - host["host_read_started"]) * 1000)
        for view in views:
            entry = host["images"][view["view_id"]]
            if (
                entry["file"] != view["file"]
                or sha256(path.parent / entry["file"]) != entry["sha256"]
            ):
                raise ValueError("PNG differs from host-observed image hash")
    if not moving_span or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Moving observations and increasing source timestamps required")
    files = (root / "result.json", rows_path, host_path, root / "control/api-receipt.json")
    pins = {str(p): sha256(p) for p in files}
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "graphics-frame cached car timestamp observed before/after each render call",
            "not_exposure_or_physics_atomicity_proof": True,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_owned_ai_frame_clock_audit",
        "frames": len(rows),
        "moving_frames": len(moving_span),
        "same_graphics_frame_count": same_frame,
        "same_car_timestamp_per_view_count": same_timestamp,
        "car_timestamp_matches_post_render_diagnostic_count": same_post,
        "car_timestamp_strictly_increases_between_groups": True,
        "moving_render_start_span_median_ms": float(np.median(moving_span)),
        "moving_render_start_span_max_ms": max(moving_span),
        "host_hash_time_median_ms": float(np.median(hash_duration)),
        "graphics_state_is_not_physics_rate": True,
        "exposure_sync_verified": False,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
