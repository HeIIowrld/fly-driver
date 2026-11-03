"""Apply conservative host receipt ordering to an original owned camera diagnostic."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.causal_render_body_match import select_prior_body_receipt
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_frame_native_clock_bounds import read, replay_clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result, control, receipt = (
        read(root / "result.json"),
        read(root / "control/result.json"),
        read(root / "control/api-receipt.json"),
    )
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
        raise ValueError("Exact completed diagnostic and clock receipts required")
    initial_data, final_data = read(initial_path), read(final_path)
    clock, final = replay_clock(initial_data), replay_clock(final_data)
    if (
        clock.generation != final.generation
        or final_data["exchanges"][: len(initial_data["exchanges"])] != initial_data["exchanges"]
        or len(final_data["exchanges"]) < len(initial_data["exchanges"]) + 2
    ):
        raise ValueError("Same generation with subsequent clock observations required")
    samples, seen = [], set()
    for row in read(native_path):
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Native source hash differs")
        sample = decode_physics_prefix(raw, row["received_monotonic_seconds"])
        if sample.packet_id != row["packet_id"]:
            raise ValueError("Native packet identity differs")
        if sample.packet_id not in seen:
            samples.append(sample)
            seen.add(sample.packet_id)
    receipts = np.array([s.received_monotonic_seconds for s in samples])
    pairs, counts, ages = [], {}, []
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
            or diag["generation"] != clock.generation
        ):
            raise ValueError("Original isolated camera diagnostic required")
        views = frame["views"]
        if len(views) != 4 or {v["view_id"] for v in views} != {"front", "left", "right", "rear"}:
            raise ValueError("Original four camera views required")
        source = np.array([v["render_started_clock"] for v in views])
        lower = source + clock.low - clock.host_quantization
        upper = source + clock.high + clock.host_quantization
        match = select_prior_body_receipt(receipts, lower, upper)
        moving = bool(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]) > 1)
        counts[match.reason] = counts.get(match.reason, 0) + 1
        if match.native_index is not None:
            sample = samples[match.native_index]
            ages.append(match.maximum_host_age_seconds)
            if not (
                sample.received_monotonic_seconds <= lower.min()
                and upper.max() - sample.received_monotonic_seconds <= 0.05
            ):
                raise ValueError("A selected receipt must satisfy every conservative bound")
        pairs.append(
            {
                "frame_index": row["index"],
                "moving_speed_candidate_gt_1m_s": moving,
                "reason": match.reason,
                "maximum_host_age_seconds": match.maximum_host_age_seconds,
                "native_packet_id": samples[match.native_index].packet_id
                if match.native_index is not None
                else None,
            }
        )
    if len(rows) != result["frames"] or not ages:
        raise ValueError("Complete frames and at least one conservative match required")
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
            "scope": "offline host receipt bounds; no physics timestamp or exposure equivalence",
            "max_host_age_seconds": 0.05,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    write_json(run / "pairs.json", pairs)
    moving = [p for p in pairs if p["moving_speed_candidate_gt_1m_s"]]
    out = {
        "status": "completed_causal_render_body_receipt_audit",
        "frames": len(rows),
        "moving_frames": len(moving),
        "reason_counts": counts,
        "moving_accepted": sum(p["reason"] == "prior_receipt_within_host_age" for p in moving),
        "accepted_age_ms_median": float(np.median(ages) * 1000),
        "accepted_age_ms_max": float(max(ages) * 1000),
        "pairs_sha256": sha256(run / "pairs.json"),
        "native_receipt_not_physics_sample_time": True,
        "physical_exposure_sync_verified": False,
        "not_actor_admission": True,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
