"""Bounded AC-only whole-window RGB and native physics receipt brackets; no app input."""

import argparse
import base64
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_clock import NativeGraphicsTimingReader
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_wgc import WGCACCapture
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def physics_record(sample):
    return {
        "packet_id": sample.packet_id,
        "fields": dict(sample.fields),
        "received_monotonic_seconds": sample.received_monotonic_seconds,
        "prefix_base64": base64.b64encode(sample.prefix_bytes).decode(),
        "prefix_sha256": hashlib.sha256(sample.prefix_bytes).hexdigest(),
        "source_time_seconds": None,
        "simulation_clock_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwnd", type=lambda s: int(s, 0))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--graphics-clock", action="store_true",
                        help="Record native graphics timing around each physics/image group")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.frames <= 10000
        or not np.isfinite(args.rate)
        or not 0 < args.rate <= 25
        or not 1 <= args.max_bytes <= 8 * 1024**3
    ):
        raise ValueError("Explicit finite frame/rate/storage budgets required")
    run = begin_run(
        args.output,
        {
            "arguments": vars(args),
            "scope": "raw AC-only WGC capture",
            "ui_actions_sent": 0,
            "actuator_messages_sent": 0,
            "synchronization": "host receipt bracket only; not simulation or exposure alignment",
        },
    )
    capture, physics, graphics = None, None, None
    count, total = 0, 0
    try:
        native = WindowsACWindow()
        windows = native.discover()
        if args.hwnd is None:
            if len(windows) != 1:
                raise ValueError("Exactly one existing original AC engine window required")
            hwnd = windows[0].hwnd
        else:
            hwnd = args.hwnd
        identity = native.probe(hwnd, require_foreground=False)
        write_json(run / "window.json", asdict(identity))
        if args.graphics_clock:
            graphics = NativeGraphicsTimingReader()
        first_graphics = graphics.snapshot() if graphics is not None else None
        physics = NativePhysicsReader(timeout=3.0)
        first_before = physics.receive()
        capture = WGCACCapture(hwnd, native=native)
        with (run / "captures.jsonl").open("x", encoding="utf8") as stream:
            for i in range(args.frames):
                started = time.monotonic()
                graphics_before = (first_graphics if i == 0 else graphics.snapshot()
                                   if graphics is not None else None)
                before = first_before if i == 0 else physics.receive()
                sample = capture.capture(not_before=before.received_monotonic_seconds)
                after = physics.receive()
                graphics_after = graphics.snapshot() if graphics is not None else None
                if not (
                    before.received_monotonic_seconds
                    <= sample.started_monotonic_seconds
                    <= sample.finished_monotonic_seconds
                    <= after.received_monotonic_seconds
                ):
                    raise ValueError("Host receipt bracket order differs")
                if graphics_before is not None and not (
                    graphics_before.received_monotonic_seconds
                    <= before.received_monotonic_seconds
                    <= after.received_monotonic_seconds
                    <= graphics_after.received_monotonic_seconds
                ):
                    raise ValueError("Graphics/physics host receipt order differs")
                if total + sample.rgb.nbytes + 256 > args.max_bytes:
                    raise ValueError("Raw capture storage budget exhausted")
                file = run / f"rgb-{i:06d}.npy"
                with file.open("xb") as out:
                    np.save(out, sample.rgb, allow_pickle=False)
                total += file.stat().st_size
                row = sample.metadata()
                row.update(
                    file=file.name,
                    file_sha256=sha256(file),
                    was_foreground_at_receipt=bool(native.user.GetForegroundWindow() == hwnd),
                    physics_before=physics_record(before),
                    physics_after=physics_record(after),
                    body_image_synchronization_verified=False,
                )
                if graphics_before is not None:
                    row.update(
                        graphics_before=graphics_before.metadata(),
                        graphics_after=graphics_after.metadata(),
                        graphics_readings_are_exposure_bounds=False,
                        native_time_mapping_verified=False,
                    )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                count += 1
                if i + 1 < args.frames:
                    time.sleep(max(0, 1 / args.rate - (time.monotonic() - started)))
        finished_capture, capture = capture, None
        finished_capture.close()
        physics.close()
        physics = None
        if graphics is not None:
            graphics.close()
            graphics = None
        write_json(
            run / "result.json",
            {
                "status": "completed",
                "frames": count,
                "bytes": total,
                "ui_actions_sent": 0,
                "actuator_messages_sent": 0,
                "actor_stream_created": False,
                "game_render_freshness_verified": False,
                "body_image_synchronization_verified": False,
                "graphics_timing_recorded": args.graphics_clock,
                "goal_achieved": False,
            },
        )
    except Exception as error:
        if capture is not None:
            write_json(run / "capture-diagnostics.json", capture.diagnostics())
        write_json(
            run / "result.json",
            {
                "status": "failed",
                "frames": count,
                "error": type(error).__name__,
                "message": str(error),
                "ui_actions_sent": 0,
                "actuator_messages_sent": 0,
                "goal_achieved": False,
            },
        )
        raise
    finally:
        if graphics is not None:
            graphics.close()
        if physics is not None:
            physics.close()
        if capture is not None:
            try:
                capture.close()
            except Exception as error:
                write_json(
                    run / "cleanup-error.json",
                    {"error": type(error).__name__, "message": str(error)},
                )


if __name__ == "__main__":
    main()
