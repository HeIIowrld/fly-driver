"""Bounded raw AC-window capture with acquisition metadata; no actor or control actions."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from flybrain_drive.ac_capture import MSSACCapture, WindowsACWindow
from flybrain_drive.evidence import begin_run, write_json


def record_frames(capture, run, *, frames, rate, max_bytes):
    if (type(frames) is not int or not 1 <= frames <= 10000
            or not np.isfinite(rate) or not 0 < rate <= 25
            or type(max_bytes) is not int or not 1 <= max_bytes <= 8 * 1024**3):
        raise ValueError("Finite frame, rate and storage budgets required")
    count, total = 0, 0
    with (run / "captures.jsonl").open("x", encoding="utf8") as stream:
        for index in range(frames):
            started = time.monotonic()
            sample = capture.capture()
            if sample.capture_id != index:
                raise ValueError("Capture stream reset or duplicate")
            # NPY overhead is below 256 bytes for this bounded uint8 HWC array.
            if total + sample.rgb.nbytes + 256 > max_bytes:
                raise ValueError("Capture storage budget exhausted")
            path = run / ("rgb-%06d.npy" % index)
            with path.open("xb") as out:
                np.save(out, sample.rgb, allow_pickle=False)
            total += path.stat().st_size
            metadata = sample.metadata()
            metadata.update(file=path.name,
                            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            stream.write(json.dumps(metadata, allow_nan=False) + "\n")
            stream.flush()
            count += 1
            if index + 1 < frames:
                time.sleep(max(0., 1/rate - (time.monotonic() - started)))
    return {"frames": count, "bytes": total, "actor_stream_created": False,
            "game_runtime_verified": False,
            "game_render_freshness_verified": False, "body_synchronization_verified": False,
            "actuator_messages_sent": 0, "goal_achieved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-windows", action="store_true")
    mode.add_argument("--hwnd", type=lambda value: int(value, 0))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--rate", type=float, default=10.)
    parser.add_argument("--max-bytes", type=int, default=512*1024**2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = begin_run(args.output, {"arguments": vars(args), "scope": "read-only raw AC capture",
                                 "actor_stream_created": False, "actuator_messages_sent": 0})
    capture = None
    try:
        if args.list_windows:
            windows = [asdict(state) for state in WindowsACWindow().discover()]
            write_json(run / "windows.json", windows)
            result = {"windows": len(windows), "pixels_acquired": 0,
                      "game_runtime_verified": False, "actuator_messages_sent": 0}
        else:
            capture = MSSACCapture(args.hwnd)
            result = record_frames(capture, run, frames=args.frames, rate=args.rate,
                                   max_bytes=args.max_bytes)
        write_json(run / "result.json", {"status": "completed", **result})
    except Exception as error:
        index = run / "captures.jsonl"
        count = len(index.read_text(encoding="utf8").splitlines()) if index.exists() else 0
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error), "actor_stream_created": False,
                                         "capture_index_lines": count,
                                         "actuator_messages_sent": 0})
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    main()
