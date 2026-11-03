"""Stage original four-view retina and prior native body samples for diagnostic replay."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.causal_render_body_match import select_prior_body_receipt
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import decode_navigation_body_prefix
from scripts.audit_frame_native_clock_bounds import read, replay_clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--pixel-archive", required=True)
    parser.add_argument("--causal-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, archive, audit = map(Path, (args.recording, args.pixel_archive, args.causal_audit))
    result, archive_result, proof = (
        read(root / "result.json"),
        read(archive / "result.json"),
        read(audit / "result.json"),
    )
    if (
        result["status"] != "completed"
        or archive_result["status"] != "completed_diagnostic_owned_ai_image_archive_and_projection"
        or not archive_result["host_observed_png_hashes_verified"]
        or archive_result["training_admissible"] is not False
        or proof["status"] != "completed_causal_render_body_receipt_audit"
        or sha256(audit / "pairs.json") != proof["pairs_sha256"]
        or sha256(archive / "retinal-inputs.npz") != archive_result["retinal_sha256"]
        or sha256(root / "diagnostic-frames.json") != result["metadata_sha256"]
        or sha256(root / "clock-initial.json") != result["clock_initial_sha256"]
    ):
        raise ValueError("Original completed isolated vision, clock and body audit required")
    archive_manifest = read(archive / "manifest.json")["config"]["source_hashes"]
    audit_manifest = read(audit / "manifest.json")["config"]["source_hashes"]
    for manifest in (archive_manifest, audit_manifest):
        for path, digest in manifest.items():
            if sha256(path) != digest:
                raise ValueError("Original retained input differs from source manifest")
    with np.load(archive / "retinal-inputs.npz", allow_pickle=False) as data:
        retinal = data["retinal"].copy()
        source_clocks = data["render_source_clocks"].copy()
        rgb_sha = data["rgb_sha256"].copy()
    rows, pairs = read(root / "diagnostic-frames.json"), read(audit / "pairs.json")
    if (
        retinal.shape != (result["frames"], 4, 721)
        or retinal.dtype != np.float32
        or source_clocks.shape != (result["frames"], 4)
        or rgb_sha.shape != source_clocks.shape
        or len(rows) != result["frames"]
        or len(pairs) != len(rows)
        or not np.isfinite(retinal).all()
        or not np.isfinite(source_clocks).all()
    ):
        raise ValueError("Complete original four-view retina and clocks required")
    clock = replay_clock(read(root / "clock-initial.json"))
    original_samples, seen = [], set()
    for row in read(root / "control/native-motion-samples.json"):
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Original native prefix hash differs")
        sample = decode_navigation_body_prefix(raw, row["received_monotonic_seconds"])
        if sample.legacy.packet_id != row["packet_id"]:
            raise ValueError("Original native packet ID differs")
        if sample.legacy.packet_id not in seen:
            original_samples.append(sample)
            seen.add(sample.legacy.packet_id)
    receipts = np.array([s.legacy.received_monotonic_seconds for s in original_samples])
    body = np.zeros((len(rows), 21), dtype=np.float32)
    available = np.zeros(len(rows), dtype=bool)
    packet_ids = np.full(len(rows), -1, dtype=np.int64)
    body_receipts = np.zeros(len(rows), dtype=np.float64)
    age_upper = np.zeros(len(rows), dtype=np.float64)
    views = ("front", "left", "right", "rear")
    for index, (row, prior) in enumerate(zip(rows, pairs, strict=True)):
        path = Path(row["source_file"])
        if row["index"] != index or prior["frame_index"] != index or sha256(path) != row["sha256"]:
            raise ValueError("Original per-frame identity/hash differs")
        frame = read(path)
        if (
            frame["request_id"] != result["request_id"]
            or not frame["aid_probe_only"]
            or frame["airflow_diagnostic"]["generation"] != clock.generation
        ):
            raise ValueError("Original same-generation diagnostic required")
        original_views = {x["view_id"]: x for x in frame["views"]}
        if set(original_views) != set(views) or any(
            original_views[name]["render_started_clock"] != source_clocks[index, j]
            for j, name in enumerate(views)
        ):
            raise ValueError("Archive retinal clocks differ from original camera metadata")
        lower = source_clocks[index] + clock.low - clock.host_quantization
        upper = source_clocks[index] + clock.high + clock.host_quantization
        match = select_prior_body_receipt(receipts, lower, upper)
        packet = (
            original_samples[match.native_index].legacy.packet_id
            if match.native_index is not None
            else None
        )
        if match.reason != prior["reason"] or packet != prior["native_packet_id"]:
            raise ValueError("Receipt selection differs from independently saved audit")
        if match.native_index is None:
            continue
        sample = original_samples[match.native_index]
        native = sample.legacy
        # Raw native conventions only: no world heading, pose, map or teacher action.
        body[index] = [
            native.pitch_native,
            native.roll_native,
            *native.angular_native,
            *native.acceleration_native_g,
            *native.wheel_load_native,
            *native.tyre_temperature_native,
            sample.speed_magnitude_m_s,
            *sample.wheel_rotation_native_rad_s,
        ]
        available[index], packet_ids[index] = True, packet
        body_receipts[index] = native.received_monotonic_seconds
        age_upper[index] = match.maximum_host_age_seconds
    if (
        not np.isfinite(body).all()
        or int(available.sum()) != proof["reason_counts"]["prior_receipt_within_host_age"]
    ):
        raise ValueError("Finite body and exact original causal count required")
    inputs = [
        root / "result.json",
        root / "diagnostic-frames.json",
        root / "clock-initial.json",
        root / "control/native-motion-samples.json",
        archive / "result.json",
        archive / "retinal-inputs.npz",
        audit / "result.json",
        audit / "pairs.json",
    ]
    pins = {str(p): sha256(p) for p in inputs}
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "causal host-receipt staging only; raw native axes not actor contract",
            "no_map_world_pose_teacher_inputs": True,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(
        run / "diagnostic-inputs.npz",
        retina=retinal,
        native_unregistered_body=body,
        native_feature_names=np.array(
            [
                "pitch_native_rad",
                "roll_native_rad",
                "angular_native_0",
                "angular_native_1",
                "angular_native_2",
                "accG_native_0",
                "accG_native_1",
                "accG_native_2",
                *[f"wheel_load_native_{i}" for i in range(4)],
                *[f"tyre_temperature_native_{i}" for i in range(4)],
                "speed_magnitude_m_s",
                *[f"wheel_rotation_native_{i}" for i in range(4)],
            ]
        ),
        native_receipt_available=available,
        native_packet_id=packet_ids,
        native_receipt_host_seconds=body_receipts,
        body_max_host_age_seconds=age_upper,
        render_source_clocks=source_clocks,
        rgb_sha256=rgb_sha,
    )
    out = {
        "status": "completed_causal_fourview_body_diagnostic_stage",
        "frames": len(rows),
        "receipt_matched": int(available.sum()),
        "unmatched": int((~available).sum()),
        "unmatched_body_is_zero_with_false_mask": bool(np.all(body[~available] == 0)),
        "native_axes_are_unregistered_not_actor_ready": True,
        "source_time_or_physical_exposure_sync_verified": False,
        "flyvis_executed": False,
        "training_admissible": False,
        "actor_deployed": False,
        "diagnostic_inputs_sha256": sha256(run / "diagnostic-inputs.npz"),
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
