"""Evaluate causal receipt-window yaw against diagnostic pose changes offline."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.causal_body_window import prior_window_mean
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
    meta, initial, packets = (root / "diagnostic-frames.json", root / "clock-initial.json",
                              root / "control/native-motion-samples.json")
    if (result["status"] != "completed" or sha256(meta) != result["metadata_sha256"]
            or sha256(initial) != result["clock_initial_sha256"]):
        raise ValueError("Completed original recording and clock required")
    clock = replay_clock(read(initial))
    times, yaw, seen = [], [], set()
    for row in read(packets):
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Native prefix hash changed")
        sample = decode_physics_prefix(raw, row["received_monotonic_seconds"])
        if sample.packet_id != row["packet_id"]:
            raise ValueError("Native packet identity changed")
        if sample.packet_id not in seen:
            times.append(sample.received_monotonic_seconds)
            yaw.append(sample.fields["localAngularVel"][1])
            seen.add(sample.packet_id)
    times, yaw = np.asarray(times), np.asarray(yaw)
    frames = []
    for i, row in enumerate(read(meta)):
        path = Path(row["source_file"])
        if i != row["index"] or sha256(path) != row["sha256"]:
            raise ValueError("Original frame order and hash required")
        frame = read(path)
        if (frame["airflow_diagnostic"] != row["diagnostic"]
                or frame["generation"] != clock.generation):
            raise ValueError("Same-generation diagnostic metadata required")
        frames.append(frame)
    if len(frames) != result["frames"]:
        raise ValueError("Complete diagnostic sequence required")
    diag = [x["airflow_diagnostic"] for x in frames]
    clock_times = np.asarray([x["started_source_clock"] for x in diag])
    directions = np.asarray([x["body_forward_candidate"] for x in diag])
    speeds = np.asarray([np.linalg.norm(x["vehicle_velocity_candidate_m_s"]) for x in diag])
    heading = np.unwrap(np.arctan2(directions[:, 0], directions[:, 2]))
    dt = np.diff(clock_times)
    if np.any(dt <= 0):
        raise ValueError("Increasing source times required")
    rows, reasons = [], {}
    for i, span in enumerate(dt):
        if not (span <= .15 and speeds[i] > 1 and speeds[i+1] > 1):
            continue
        start_lower = min(x["render_started_clock"] for x in frames[i+1]["views"])
        start_upper = max(x["render_started_clock"] for x in frames[i+1]["views"])
        end = start_lower + clock.low - clock.host_quantization
        upper = start_upper + clock.high + clock.host_quantization
        estimate = prior_window_mean(times, yaw, end_lower=end, end_upper=upper,
                                     duration=float(span))
        reasons[estimate.reason] = reasons.get(estimate.reason, 0) + 1
        if estimate.value is None:
            continue
        latest_index = np.searchsorted(times, end, side="left")-1
        rows.append((i, (heading[i+1]-heading[i])/span, estimate.value, yaw[latest_index]))
    if not rows:
        raise ValueError("No valid causal moving windows")
    data = np.asarray(rows)
    target = data[:, 1]
    metrics = {}
    for label, pred in (("window", data[:, 2]), ("latest", data[:, 3])):
        metrics[label] = {"rmse": float(np.sqrt(np.mean((pred-target)**2))),
                          "correlation": float(np.corrcoef(pred, target)[0, 1])}
        for name, select in (("positive", target > .02), ("negative", target < -.02)):
            metrics[label][name] = {"count": int(select.sum()),
                "rmse": float(np.sqrt(np.mean((pred[select]-target[select])**2)))
                if select.any() else None}
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "sensor_source_sha256": sha256("src/flybrain_drive/causal_body_window.py"),
        "source_hashes": {str(p): sha256(p) for p in
                          (root / "result.json", meta, initial, packets)},
        "world_heading_diagnostic_only": True, "future_receipts_excluded": True})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / "predictions.npz", frame_index=data[:, 0].astype(np.int64),
                        target=data[:, 1], causal_window=data[:, 2], causal_latest=data[:, 3])
    report = {"status": "completed_causal_body_window_diagnostic", "windows": len(rows),
              "reasons": reasons, "metrics": metrics,
              "predictions_sha256": sha256(run / "predictions.npz"),
              "physical_sample_or_exposure_sync_verified": False,
              "native_axis_registered_by_this_test": False, "actor_deployed": False,
              "optimizer_updates": 0}
    write_json(run / "result.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
