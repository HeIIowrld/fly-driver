"""Read-only teacher SDK cadence while recording four-view CSP frames; no control."""
import json
import os
import statistics
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

PIN = '78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6'


def main():
    output = 'D:/CodexResearchRuns/flybrain-teacher-cadence-camera-20261102-01'
    run = begin_run(output, dict(script_sha256=sha256(__file__), camera_pin=PIN,
                                 duration_seconds=6, control=False))
    rows, camera = [], None
    status = NativeTeacherStatusReader()
    try:
        with NativePhysicsReader(timeout=.5) as native:
            sample = native.receive()
            if (status.snapshot()['builtin_ai_controlled'] is not False or
                    abs(sample.fields['speedKmh']) >= .1 or sample.fields['brake'] <= .98):
                raise ValueError('Stationary offline neutral car required')
        window = WindowsACWindow().discover()
        if len(window) != 1:
            raise ValueError('Exactly one AC engine required')
        config = json.loads(Path('configs/ac-csp-retina.json').read_text(encoding='utf8'))
        retina = CSPCameraRetina(config)
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                     timeout=.2, poll_seconds=.002, clock=time.perf_counter)
        camera = CSPVisionRecorder(window[0].hwnd, config, max_frames=300,
            pixel_preprocessor=retina.precompute_pixels, expected_source_sha256=PIN)
        camera.wait_ready()
        deadline = time.perf_counter() + 6
        while time.perf_counter() < deadline:
            p = client.receive()
            rows.append(dict(sequence=p.sequence, source=p.source_monotonic_seconds,
                             host=p.received_monotonic_seconds))
        gaps = [b['source'] - a['source'] for a, b in zip(rows, rows[1:])]
        result = dict(status='completed', count=len(rows), source_median_ms=statistics.median(gaps)*1000,
                      source_max_ms=max(gaps)*1000, source_gt80ms=sum(x > .08 for x in gaps),
                      sequence_skips=sum(max(0,b['sequence']-a['sequence']-1)
                                         for a,b in zip(rows,rows[1:])),
                      actuator_messages_sent=0, training_admissible=False)
    except Exception as e:
        result = dict(status='failed', error=type(e).__name__, reason=str(e),
                      actuator_messages_sent=0, training_admissible=False)
    finally:
        if camera is not None:
            try:
                result['camera'] = camera.close()
                result['archive'] = camera.archive_to(run / 'vision', bundle=True)
            except Exception as e:
                result['camera_cleanup_error'] = str(e)
        status.close()
        write_json(run / 'samples.json', rows)
        write_json(run / 'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('camera','archive')}))


if __name__ == '__main__':
    main()
