"""Bounded read-only AC image/physics cadence probe; stores endpoint images only."""

import argparse
import json
import time

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_wgc import WGCACCapture
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--rate', type=float, default=10)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 2 <= args.frames <= 1000 or not np.isfinite(args.rate) or not 1 <= args.rate <= 25:
        raise ValueError('Finite frame count and rate required')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
                                     ui_actions_sent=0, actuator_messages_sent=0))
    native = WindowsACWindow()
    capture = physics = None
    rows = []
    try:
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine window required')
        hwnd = windows[0].hwnd
        physics = NativePhysicsReader(timeout=3)
        first = physics.receive()
        capture = WGCACCapture(hwnd, native=native)
        with (run / 'samples.jsonl').open('x', encoding='utf8') as stream:
            for i in range(args.frames):
                started = time.monotonic()
                before = first if i == 0 else physics.receive()
                sample = capture.capture(not_before=before.received_monotonic_seconds)
                after = physics.receive()
                if not (before.received_monotonic_seconds <= sample.started_monotonic_seconds
                        <= sample.finished_monotonic_seconds <= after.received_monotonic_seconds):
                    raise ValueError('Host receipt order differs')
                row = sample.metadata()
                row.update(
                    was_foreground_at_receipt=bool(native.user.GetForegroundWindow() == hwnd),
                    physics_before=physics_record(before), physics_after=physics_record(after),
                    file=None, file_sha256=None)
                if i in (0, args.frames - 1):
                    path = run / f'rgb-{i:06d}.npy'
                    with path.open('xb') as out:
                        np.save(out, sample.rgb, allow_pickle=False)
                    row.update(file=path.name, file_sha256=sha256(path))
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                if i + 1 < args.frames:
                    time.sleep(max(0, 1 / args.rate - (time.monotonic() - started)))
        stamps = np.array([r['finished_monotonic_seconds'] for r in rows])
        gaps = np.diff(stamps)
        write_json(run / 'result.json', dict(
            status='completed', frames=len(rows), duration_seconds=float(stamps[-1]-stamps[0]),
            received_hz=float((len(rows)-1)/(stamps[-1]-stamps[0])),
            receipt_gap_seconds=dict(p50=float(np.percentile(gaps, 50)),
                                     p95=float(np.percentile(gaps, 95)), max=float(gaps.max())),
            foreground_frames=sum(r['was_foreground_at_receipt'] for r in rows),
            unique_rgb_hashes=len({r['rgb_sha256'] for r in rows}),
            capture_diagnostics=capture.diagnostics(), ui_actions_sent=0, actuator_messages_sent=0,
            exposure_clock_verified=False, moving_driving_verified=False,
            actor_stream_created=False, goal_achieved=False))
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', frames=len(rows),
            error=type(error).__name__, message=str(error),
            capture_diagnostics=capture.diagnostics() if capture is not None else None,
            ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))
        raise
    finally:
        if physics is not None:
            physics.close()
        if capture is not None:
            try:
                capture.close()
            except Exception as error:
                write_json(run / 'cleanup-error.json', dict(error=type(error).__name__,
                                                           message=str(error)))


if __name__ == '__main__':
    main()
