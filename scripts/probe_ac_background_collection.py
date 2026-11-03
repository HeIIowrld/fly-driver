"""Read-only baseline/async-image collection timing; no vJoy, keys or focus changes."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_vision import TeacherVisionRecorder
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-per-phase', type=int, default=40)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 10 <= args.samples_per_phase <= 100:
        raise ValueError('Finite 10..100 sample timing phases required')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, period_seconds=.04,
        ui_actions_sent=0, actuator_messages_sent=0, teacher_inferences=0))
    vision = None
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
                  training_admissible=False, goal_achieved=False)
    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC window required')
        window = windows[0]
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                     timeout=.2)
        phases = {}
        result['phases'] = phases
        with NativePhysicsReader(timeout=.2) as physics:
            with (run / 'sensor-image-candidates.jsonl').open('x', encoding='utf8') as stream:
                for phase in ('baseline', 'async_vision'):
                    if phase == 'async_vision':
                        vision = TeacherVisionRecorder(run / 'vision', window.hwnd,
                            max_frames=150, max_bytes=768 * 1024**2)
                        vision.wait_ready()
                    elapsed, foreground, matches = [], 0, 0
                    phase_start = time.monotonic()
                    for index in range(args.samples_per_phase):
                        started = time.monotonic()
                        packet, sample = client.receive(), physics.receive()
                        candidate, missing = None, None
                        if vision is not None:
                            try:
                                candidate = vision.candidate_before(min(
                                    packet.received_monotonic_seconds,
                                    sample.received_monotonic_seconds))
                                matches += 1
                            except ValueError as error:
                                missing = str(error)
                        is_foreground = bool(native.user.GetForegroundWindow() == window.hwnd)
                        foreground += is_foreground
                        stream.write(json.dumps(dict(phase=phase, index=index,
                            sdk=record_packet(packet), native=physics_record(sample),
                            image_candidate=candidate, image_unavailable_reason=missing,
                            was_foreground_at_record=is_foreground), allow_nan=False) + '\n')
                        stream.flush()
                        elapsed.append(time.monotonic() - started)
                        time.sleep(max(0, .04 - elapsed[-1]))
                    duration = time.monotonic() - phase_start
                    phases[phase] = dict(samples=args.samples_per_phase,
                        duration_seconds=duration,
                        receipt_loop_hz=args.samples_per_phase / duration,
                        work_seconds_percentiles=np.percentile(elapsed, [50, 95, 99]).tolist(),
                        foreground_samples=foreground, image_candidates=matches)
        result.update(status='completed', phases=phases)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        if vision is not None:
            result['vision'] = vision.close()
        write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
