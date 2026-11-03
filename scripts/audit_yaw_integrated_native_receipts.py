"""Offline interval integration of retained native yaw receipts against camera pose."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_frame_native_clock_bounds import replay_clock


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result = read(root / "result.json")
    clock_path, final_path = root / "clock-initial.json", root / "clock-final.json"
    metadata, samples = root / "diagnostic-frames.json", root / "control/native-motion-samples.json"
    if (result["status"] != "completed" or sha256(metadata) != result["metadata_sha256"]
            or sha256(clock_path) != result["clock_initial_sha256"]
            or sha256(final_path) != result["clock_final_sha256"]):
        raise ValueError("Completed original recording and both clock receipts required")
    initial, final = read(clock_path), read(final_path)
    clock = replay_clock(initial)
    replay_clock(final)
    if (final["generation"] != initial["generation"]
            or final["exchanges"][:len(initial["exchanges"])] != initial["exchanges"]):
        raise ValueError("Same-generation clock observations required")
    packet_times, yaw, seen = [], [], set()
    for row in read(samples):
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Native bytes differ from original receipt")
        sample = decode_physics_prefix(raw, row["received_monotonic_seconds"])
        if sample.packet_id != row["packet_id"]:
            raise ValueError("Native packet identity changed")
        if sample.packet_id not in seen:
            packet_times.append(sample.received_monotonic_seconds)
            yaw.append(sample.fields["localAngularVel"][1])
            seen.add(sample.packet_id)
    packet_times, yaw = np.asarray(packet_times), np.asarray(yaw)
    if (len(yaw) < 3 or not np.isfinite(packet_times).all() or not np.isfinite(yaw).all()
            or np.any(np.diff(packet_times) <= 0)):
        raise ValueError("Ordered, finite, unique native packets required")
    times, directions, speeds = [], [], []
    rows = read(metadata)
    for i, row in enumerate(rows):
        path = Path(row["source_file"])
        if row["index"] != i or sha256(path) != row["sha256"]:
            raise ValueError("Original diagnostic frame and order required")
        frame = read(path)
        diag = frame["airflow_diagnostic"]
        if diag != row["diagnostic"] or diag["generation"] != clock.generation:
            raise ValueError("Original same-generation diagnostic required")
        times.append(diag["started_source_clock"])
        directions.append(diag["body_forward_candidate"])
        speeds.append(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]))
    times, directions, speeds = map(np.asarray, (times, directions, speeds))
    if (len(rows) != result["frames"] or directions.shape != (len(rows), 3)
            or not all(np.isfinite(x).all() for x in (times, directions, speeds))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Finite ordered frames required")
    heading = np.unwrap(np.arctan2(directions[:, 0], directions[:, 2]))
    offsets = {"lower": clock.low, "midpoint": (clock.low+clock.high)/2, "upper": clock.high}
    predictions, targets, indices = {k: [] for k in offsets}, [], []
    max_gap = .05
    for i, dt in enumerate(np.diff(times)):
        if not (0 < dt <= .15 and speeds[i] > 1 and speeds[i+1] > 1):
            continue
        windows = {}
        for label, offset in offsets.items():
            start, end = times[i]+offset, times[i+1]+offset
            left = np.searchsorted(packet_times, start, side="right")-1
            right = np.searchsorted(packet_times, end, side="left")
            if left < 0 or right >= len(packet_times):
                break
            ts, ys = packet_times[left:right+1], yaw[left:right+1]
            if np.max(np.diff(ts)) > max_gap:
                break
            inner = (ts > start) & (ts < end)
            grid = np.r_[start, ts[inner], end]
            values = np.interp(grid, ts, ys)
            windows[label] = float(np.trapezoid(values, grid) / dt)
        if len(windows) == len(offsets):
            indices.append(i)
            targets.append((heading[i+1]-heading[i])/dt)
            for label in offsets:
                predictions[label].append(windows[label])
    targets = np.asarray(targets)
    if len(targets) < 3:
        raise ValueError("Sufficient fully bracketed native receipt intervals required")
    metrics = {}
    for label, values in predictions.items():
        values = np.asarray(values)
        metrics[label] = {"rmse": float(np.sqrt(np.mean((values-targets)**2))),
                          "correlation": float(np.corrcoef(values, targets)[0, 1])}
        for name, select in (("positive", targets > .02), ("negative", targets < -.02)):
            metrics[label][name] = {"count": int(select.sum()),
                "rmse": float(np.sqrt(np.mean((values[select]-targets[select])**2)))
                if select.any() else None}
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "source_hashes": {str(p): sha256(p) for p in
                          (root / "result.json", metadata, samples, clock_path, final_path)},
        "offline_only": True, "future_packets_used_for_diagnostic_interpolation": True,
        "maximum_native_receipt_gap_seconds": max_gap})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / "intervals.npz", frame_index=np.asarray(indices),
                        azimuth_rate=targets, **{k: np.asarray(v) for k, v in predictions.items()})
    report = {"status": "completed_yaw_receipt_integration", "intervals": len(targets),
        "metrics": metrics, "intervals_sha256": sha256(run / "intervals.npz"),
        "receipt_time_not_physics_sample_time": True, "physical_sync_verified": False,
        "actor_deployed": False, "optimizer_updates": 0}
    write_json(run / "result.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
