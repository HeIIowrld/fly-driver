"""Record stationary live native copy windows alongside four-view render calls.

This diagnostic does not actuate or change lifecycle; host receipts and render
calls do not establish the physical exposure or simulator sample timestamps.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_clock import CSPClockBounds, clock_ping, initialize_clock
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_physics import timed_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-camera'
    session_path = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-session/status.json'
    refuse_leased_transport('stationary live receipt audit', status_path=session_path)
    state, modified = read_published_json(session_path)
    if (not 0 <= time.time() - modified <= 2 or state.get('paused') is not False
            or state.get('ai_controlled') is not False or state.get('camera_active')
            or state.get('is_online') or state.get('in_menu')
            or abs(state.get('speed_kmh', 1)) >= .1 or state.get('gas', 1) >= .02
            or state.get('brake', 0) <= .98):
        raise ValueError('Live offline stationary full-brake non-AI session required')
    api = WindowsACWindow()
    windows = api.discover()
    if (len(windows) != 1 or api.user.GetForegroundWindow() == windows[0].hwnd
            or state['window_handle'] != str(windows[0].hwnd) + 'ULL'):
        raise ValueError('One background source engine required')
    camera, camera_modified = read_published_json(root / 'status.json')
    if (not 0 <= time.time() - camera_modified <= 3
            or camera.get('state') not in ('idle', 'completed')
            or camera.get('generation') != state['generation']):
        raise ValueError('Fresh same-generation inactive camera required')
    if camera['state'] == 'completed':
        request_id = camera.get('request_id')
        if (not isinstance(request_id, str) or not request_id.startswith('probe-')
                or len(request_id) != 38 or not all(c in '0123456789abcdef'
                                                     for c in request_id[6:])):
            raise ValueError('Exact prior owned camera request required')
        terminal = json.loads((root / request_id / 'result.json').read_text(encoding='utf8'))
        if terminal.get('status') != 'completed':
            raise ValueError('Prior camera terminal evidence required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'generation': state['generation'], 'engine_pid': windows[0].pid,
        'host_clock': 'time.perf_counter', 'role': 'diagnostic, not training',
        'physical_exposure_or_sample_sync_verified': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'actor_deployed': False, 'lifecycle_requests_sent': 0,
              'physical_exposure_or_sample_sync_verified': False}
    child = None
    try:
        clock = CSPClockBounds(state['generation'], maximum_width=.1)
        initialize_clock(root, clock, windows[0].hwnd, host_clock=time.perf_counter)
        write_json(run / 'clock-before.json', clock.record())
        cmd = [sys.executable, '-m', 'scripts.probe_ac_csp_camera', '--frames', '2',
               '--output', str(run / 'camera-probe')]
        with (run / 'camera.log').open('x', encoding='utf8') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            result['child_pid'] = child.pid
            write_json(run / 'progress.json', result)
            samples = []
            deadline = time.monotonic() + 45
            with ExistingNamedMapping('Local\\acpmf_physics', AI_STATUS_PREFIX.size) as mapping:
                try:
                    next_status = 0.
                    while child.poll() is None and time.monotonic() < deadline:
                        if len(samples) >= 20000:
                            raise ValueError('Native sampling budget exhausted; inspect child')
                        now = time.perf_counter()
                        if now >= next_status:
                            current, modified = read_published_json(session_path)
                            if (not 0 <= time.time()-modified <= 2
                                    or current.get('generation') != state['generation']
                                    or current.get('paused') is not False
                                    or current.get('is_online') is not False
                                    or current.get('ai_controlled') is not False
                                    or current.get('expert_probe_active') is not False):
                                raise ValueError('Same-generation offline AI-off state required')
                            next_status = now+.1
                        sample, timing = timed_teacher_snapshot(mapping)
                        fields = sample.physics.fields
                        samples.append({'authority': sample.authority_record(), 'copy': timing})
                        if (sample.builtin_ai_controlled or abs(fields['speedKmh']) >= .1
                                or fields['gas'] >= .02 or fields['brake'] <= .98):
                            raise ValueError('Native AI off and stationary full brake required')
                        time.sleep(.002)
                finally:
                    write_json(run / 'native-receipts.json', samples)
            if child.poll() is None:
                result.update(status='child_outcome_unknown', child_still_live=True)
                raise TimeoutError('Original child still live; never retry')
            if child.returncode != 0:
                raise RuntimeError(f'Camera child failed with exit {child.returncode}')
        for _ in range(4):
            clock_ping(root, clock, windows[0].hwnd, host_clock=time.perf_counter)
        write_json(run / 'clock-after.json', clock.record())
        frames = []
        for file in sorted((run / 'camera-probe/camera').glob('frame-*.json')):
            frame = json.loads(file.read_text(encoding='utf8'))
            if frame['generation'] != state['generation']:
                raise ValueError('Camera generation changed')
            for view in frame['views']:
                t = view['render_started_clock']
                lo, hi = t + clock.low, t + clock.high
                earlier = [x for x in samples if x['copy']['second_copy_end'] <= lo]
                later = [x for x in samples if x['copy']['first_copy_start'] >= hi]
                frames.append({'frame': frame['index'], 'view': view['view_id'],
                    'render_host_bounds': [lo, hi],
                    'last_earlier_copy_end': earlier[-1]['copy']['second_copy_end']
                    if earlier else None,
                    'last_earlier_packet': earlier[-1]['authority']['packet_id']
                    if earlier else None,
                    'first_later_copy_start': later[0]['copy']['first_copy_start']
                    if later else None, 'first_later_packet': later[0]['authority']['packet_id']
                    if later else None})
        if len(frames) != 8 or len(samples) < 2:
            raise ValueError('Two four-view groups and advancing native receipts required')
        result.update(status='completed_live_camera_native_copy_window_audit', frames=frames,
            native_receipt_count=len(samples),
            native_receipt_sha256=sha256(run / 'native-receipts.json'),
            clock_uncertainty_seconds=clock.high-clock.low,
            native_receipt_is_not_physics_sample_time=True,
            camera_render_is_not_exposure_time=True, camera_training_admissible=False,
            child_exit_code=child.returncode)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        if child is not None:
            result['child_still_live'] = child.poll() is None
            result['child_pid'] = child.pid
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
