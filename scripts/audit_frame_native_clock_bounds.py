"""Compare camera render-start bounds with original native receipt timestamps."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def replay_clock(data):
    clock = CSPClockBounds(
        data["generation"],
        maximum_width=data["maximum_width_seconds"],
        host_quantization=data.get("host_clock_quantization_seconds", 0),
    )
    for p in data["exchanges"]:
        clock.observe(p["sent"], p["received"], p["producer_clock"], data["generation"])
    if clock.record() != data:
        raise ValueError("Original clock bounds must replay exactly")
    return clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result, control = read(root / "result.json"), read(root / "control/result.json")
    receipt = read(root / "control/api-receipt.json")
    initial_path, final_path = root / "clock-initial.json", root / "clock-final.json"
    rows_path, native_path = (
        root / "diagnostic-frames.json",
        root / "control/native-motion-samples.json",
    )
    if (
        result["status"] != "completed"
        or control["status"] != "completed"
        or receipt["status"] != "api_returned_true"
        or not receipt["motion_probe_passed"]
        or control["native_motion_receipt_clock"] != "host_monotonic_QPC_time.perf_counter"
        or sha256(initial_path) != result["clock_initial_sha256"]
        or sha256(final_path) != result["clock_final_sha256"]
        or sha256(rows_path) != result["metadata_sha256"]
    ):
        raise ValueError("Completed owned diagnostic with original clock receipts required")
    initial_data, final_data = read(initial_path), read(final_path)
    initial, final = replay_clock(initial_data), replay_clock(final_data)
    if (
        initial.generation != final.generation
        or final_data["exchanges"][: len(initial_data["exchanges"])] != initial_data["exchanges"]
        or len(final_data["exchanges"]) < len(initial_data["exchanges"]) + 2
    ):
        raise ValueError("Same generation and post-run independent clock observations required")
    native, seen = [], set()
    for row in read(native_path):
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Native prefix bytes differ")
        s = decode_physics_prefix(raw, row["received_monotonic_seconds"])
        if s.packet_id != row["packet_id"]:
            raise ValueError("Original packet identity differs")
        if s.packet_id not in seen:
            native.append(s)
            seen.add(s.packet_id)
    times = np.array([s.received_monotonic_seconds for s in native])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Increasing independent native receipts required")
    pairs, speed_diff = [], []
    rows = read(rows_path)
    for row in rows:
        path = Path(row["source_file"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Original camera metadata changed")
        frame = read(path)
        diag = frame["airflow_diagnostic"]
        if (
            diag != row["diagnostic"]
            or frame["request_id"] != result["request_id"]
            or not frame["aid_probe_only"]
            or not frame["deferred_save_probe"]
            or diag["generation"] != initial.generation
        ):
            raise ValueError("Original isolated matching camera diagnostic required")
        views = {v["view_id"]: v for v in frame["views"]}
        if set(views) != {"front", "left", "right", "rear"}:
            raise ValueError("Four camera directions required")
        start = views["front"]["render_started_clock"]
        low = initial.render_start_lower(start, initial.generation)
        high = start + initial.high + initial.host_quantization
        midpoint = (low + high) / 2
        index = int(np.argmin(np.abs(times - midpoint)))
        if abs(times[index] - midpoint) <= 0.05:
            sample = native[index]
            velocity = float(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]) * 3.6)
            pairs.append(
                {
                    "camera_index": row["index"],
                    "native_packet": sample.packet_id,
                    "native_receipt_minus_front_render_start_ms_bounds": [
                        float((times[index] - high) * 1000),
                        float((times[index] - low) * 1000),
                    ],
                    "camera_speed_candidate_kmh": velocity,
                    "native_speed_kmh": float(sample.fields["speedKmh"]),
                }
            )
            speed_diff.append(abs(velocity - sample.fields["speedKmh"]))
    if not pairs:
        raise ValueError("No close native receipts found; do not invent synchronization")
    files = (
        root / "result.json",
        root / "control/result.json",
        root / "control/api-receipt.json",
        initial_path,
        final_path,
        rows_path,
        native_path,
    )
    pins = {str(p): sha256(p) for p in files}
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "nearest native host receipt to front render start; not sampling time",
            "clock": "initial conservative offset, bounded round trips, assumes rate stability",
            "maximum_midpoint_pair_distance_seconds": 0.05,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    write_json(run / "pairs.json", pairs)
    p = np.array([x["native_receipt_minus_front_render_start_ms_bounds"] for x in pairs])
    out = {
        "status": "completed_frame_native_receipt_clock_bounds_audit",
        "camera_frames": len(rows),
        "unique_native_samples": len(native),
        "near_pairs": len(pairs),
        "initial_clock_width_ms": (initial.high - initial.low) * 1000,
        "final_clock_width_ms": (final.high - final.low) * 1000,
        "median_receipt_minus_front_start_bounds_ms": np.median(p, axis=0).tolist(),
        "definitely_after_render_start": int((p[:, 0] > 0).sum()),
        "definitely_before_render_start": int((p[:, 1] < 0).sum()),
        "overlapping_clock_uncertainty": int(((p[:, 0] <= 0) & (p[:, 1] >= 0)).sum()),
        "median_abs_native_speed_difference_kmh": float(np.median(speed_diff)),
        "max_abs_native_speed_difference_kmh": float(max(speed_diff)),
        "pairs_sha256": sha256(run / "pairs.json"),
        "host_receipt_is_not_physics_source_time": True,
        "nearest_pair_is_not_causal_or_atomic": True,
        "exposure_sync_verified": False,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
