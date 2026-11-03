"""Bound current CSP render-call timestamps on host clock; no driving or training.

Ping/response bounds are not physical exposure or simulator physics timestamps.
The two-frame camera probe is diagnostic only, and every original frame is retained.
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
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-camera'
    session = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-session/status.json'
    refuse_leased_transport('stationary diagnostic clock audit', status_path=session)
    status, modified = read_published_json(session)
    if (not 0 <= time.time() - modified <= 2 or status.get('paused') is not True
            or status.get('ai_controlled') is not False or status.get('camera_active')
            or status.get('is_online') or status.get('in_menu')
            or status.get('speed_kmh', 1) >= .1 or status.get('gas', 1) >= .02
            or status.get('brake', 0) <= .98):
        raise ValueError('Fresh offline paused stationary full-brake session required')
    window_api = WindowsACWindow()
    windows = window_api.discover()
    if (len(windows) != 1 or window_api.user.GetForegroundWindow() == windows[0].hwnd
            or status['window_handle'] != str(windows[0].hwnd) + 'ULL'):
        raise ValueError('Single original background engine required')
    camera_status, camera_mtime = read_published_json(root / 'status.json')
    if (not 0 <= time.time() - camera_mtime <= 3
            or camera_status.get('state') not in ('idle', 'completed')
            or camera_status.get('generation') != status['generation']):
        raise ValueError('Same-generation idle camera required')
    if camera_status['state'] == 'completed':
        request_id = camera_status.get('request_id')
        if (not isinstance(request_id, str) or not request_id.startswith('probe-')
                or len(request_id) != 38 or not all(c in '0123456789abcdef'
                                                     for c in request_id[6:])):
            raise ValueError('Exact owned terminal camera request ID required')
        terminal = json.loads((root / request_id / 'result.json').read_text(encoding='utf8'))
        if terminal.get('status') != 'completed':
            raise ValueError('Prior completed camera request must have a terminal result')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'generation': status['generation'], 'engine_pid': windows[0].pid,
        'host_clock': 'time.perf_counter', 'role': 'stationary diagnostic, not training',
        'physical_exposure_sync_verified': False, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'camera_started': False, 'actor_deployed': False,
              'physical_exposure_sync_verified': False}
    try:
        bounds = CSPClockBounds(status['generation'], maximum_width=.1)
        initialize_clock(root, bounds, windows[0].hwnd, host_clock=time.perf_counter)
        before = bounds.record()
        write_json(run / 'clock-before.json', before)
        child_args = [sys.executable, '-m', 'scripts.probe_ac_csp_camera', '--frames', '2',
                      '--output', str(run / 'camera-probe')]
        with (run / 'camera-probe.log').open('x', encoding='utf8') as log:
            child = subprocess.Popen(child_args, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            result.update(camera_started=True, child_pid=child.pid)
            write_json(run / 'progress.json', result)
            try:
                code = child.wait(timeout=45)
            except subprocess.TimeoutExpired as error:
                result.update(status='child_outcome_unknown', child_still_live=True)
                raise TimeoutError('Original camera probe still live; do not restart') from error
        if code != 0:
            raise RuntimeError(f'Camera probe exited {code}; preserve original log')
        camera = json.loads((run / 'camera-probe/result.json').read_text(encoding='utf8'))
        if camera['status'] != 'completed':
            raise ValueError('Camera probe did not complete')
        for _ in range(4):
            clock_ping(root, bounds, windows[0].hwnd, host_clock=time.perf_counter)
        after = bounds.record()
        write_json(run / 'clock-after.json', after)
        frames = []
        for path in sorted((run / 'camera-probe/camera').glob('frame-*.json')):
            frame = json.loads(path.read_text(encoding='utf8'))
            if frame['generation'] != status['generation'] or frame['clock_domain'] != (
                    'csp_os_preciseClock_unaligned'):
                raise ValueError('Original frame generation and producer clock required')
            views = []
            for view in frame['views']:
                started = view['render_started_clock']
                views.append({'view': view['view_id'], 'producer_started': started,
                    'host_start_lower': started + bounds.low,
                    'host_start_upper': started + bounds.high})
            frames.append({'frame_sha256': sha256(path), 'views': views})
        if len(frames) != 2 or any(len(x['views']) != 4 for x in frames):
            raise ValueError('Exactly two original four-view groups required')
        result.update(status='completed_stationary_render_clock_bounds', frames=frames,
            clock_before_uncertainty_seconds=before['uncertainty_seconds'],
            clock_after_uncertainty_seconds=after['uncertainty_seconds'],
            clock_extrapolation_scope='only the short ping-to-ping interval',
            physical_exposure_sync_verified=False, simulator_physics_time_verified=False,
            camera_training_admissible=False, child_exit_code=code)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
