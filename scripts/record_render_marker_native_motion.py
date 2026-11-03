"""Bounded owned AI motion with camera airflow metadata, diagnostic only."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_clock import CSPClockBounds, initialize_clock, clock_ping
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    local = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    camera_root = local/'csp-camera'
    status = json.loads((camera_root/'status.json').read_text(encoding='utf8'))
    before, modified, _ = read_session_heartbeat(local/'csp-session/status.json')
    validate_heartbeat(before, time.time()-modified, mutating=True)
    if (status.get('state') == 'capturing' or before['paused'] is not True
            or before['is_online'] or before['camera_active'] or before['ai_controlled']
            or before['generation'] != status['generation']):
        raise ValueError('Fresh paused offline idle camera and AI required')
    windows = WindowsACWindow().discover()
    if len(windows) != 1:
        raise ValueError('Exactly one reviewed simulator required')
    deployed = Path(windows[0].executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    candidate = Path('integrations/ac_csp/candidates'
                      '/FlyBrainCamera-render-marker-motion-v1.lua')
    if sha256(deployed) != sha256(candidate):
        raise ValueError('Installed candidate differs')
    request_id = 'airflow-' + uuid.uuid4().hex
    request = dict(id=request_id, frames=300, period_seconds=.08, deadline_seconds=30,
                   exposure=.35, image_format='png', render_marker_motion=True,
                   expected_generation=before['generation'])
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), request=request,
        candidate_sha256=sha256(candidate), before=before, actor_deployed=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', request_id=request_id, actor_deployed=False,
                  ui_actions_sent=0, student_actuator_messages_sent=0)
    child, log, hash_log, sampler = None, None, None, None
    clock = CSPClockBounds(before['generation'], host_quantization=time.get_clock_info('perf_counter').resolution)
    directory = camera_root/request_id
    observed = []

    def harvest_available():
        while True:
            index = len(observed)
            path = directory/f'frame-{index:04d}.json'
            if not path.exists():
                return
            started = time.perf_counter()
            frame = read_frame = json.loads(path.read_text(encoding='utf8'))
            if (frame['index'] != index or frame['request_id'] != request_id
                    or frame.get('training_admissible') is not False
                    or frame.get('actor_stream_created') is not False
                    or frame['generation'] != before['generation']):
                raise ValueError('Original owned diagnostic frame identity required')
            if len(read_frame['views']) != 4 or {
                v['view_id'] for v in read_frame['views']
            } != {'front', 'left', 'right', 'rear'}:
                raise ValueError('All four unique view files required')
            images = {}
            for view in read_frame['views']:
                name = f'{index:04d}-{view["view_id"]}.png'
                if view['file'] != name:
                    raise ValueError('Exact owned frame filename required')
                images[view['view_id']] = {'file': name, 'sha256': sha256(directory/name)}
            record = {'index': index, 'metadata_sha256': sha256(path), 'images': images,
                      'host_read_started': started, 'host_hash_finished': time.perf_counter()}
            hash_log.write(json.dumps(record, ensure_ascii=False)+'\n')
            hash_log.flush()
            observed.append(record)
    try:
        initialize_clock(camera_root, clock, windows[0].hwnd, host_clock=time.perf_counter)
        write_json(run/'clock-before.json', clock.record())
        sampler = NativeCopySampler(lambda: ExistingNamedMapping('Local\\acpmf_physics', AI_STATUS_PREFIX.size))
        sampler.start()
        write_json(camera_root/'request.json', request)
        deadline = time.monotonic()+8
        while not (directory/'frame-0000.json').exists():
            if (directory/'result.json').exists():
                raise ValueError('Camera terminated before activation: '+
                                 (directory/'result.json').read_text(encoding='utf8'))
            if time.monotonic() >= deadline:
                raise TimeoutError('Original camera startup timeout')
            time.sleep(.02)
        frame = json.loads((directory/'frame-0000.json').read_text(encoding='utf8'))
        if (frame['request_id'] != request_id or frame.get('training_admissible') is not False
                or not all(v.get('diagnostic_clock_pair_before',{}).get('body_diagnostic') and v.get('diagnostic_clock_pair_after',{}).get('body_diagnostic') for v in frame['views'])):
            raise ValueError('Exact diagnostic camera activation required')
        hash_log = (run/'live-frame-hashes.jsonl').open('x', encoding='utf8')
        harvest_available()
        while True:
            session, session_modified, _ = read_session_heartbeat(local/'csp-session/status.json')
            validate_heartbeat(session, time.time()-session_modified, mutating=True)
            if (session.get('camera_active') is True
                    and session.get('camera_request_id') == request_id):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned camera did not appear in session heartbeat')
            time.sleep(.02)
        log = (run/'control.log').open('x', encoding='utf8')
        child = subprocess.Popen([sys.executable, '-m', 'scripts.control_ac_session',
            '--command', 'expert-motion-probe', '--execute', '--camera-request-id', request_id,
            '--expected-lua-sha256', sha256(candidate),
            '--output', str(run/'control')], stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        result['control_pid'] = child.pid
        deadline = time.monotonic()+20
        while child.poll() is None:
            harvest_available()
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned control process remains live after original deadline')
            time.sleep(.01)
        code = child.returncode
        result['control_exit_code'] = code
        if code != 0:
            raise RuntimeError('Owned AI motion failed; preserve control receipt and log')
        (directory/'stop').touch(exist_ok=False)
        deadline = time.monotonic()+5
        while not (directory/'result.json').exists():
            harvest_available()
            if time.monotonic() >= deadline:
                raise TimeoutError('Original camera stop outcome unknown')
            time.sleep(.02)
        harvest_available()
        terminal = json.loads((directory/'result.json').read_text(encoding='utf8'))
        if terminal['status'] not in ('completed', 'stopped'):
            raise ValueError('Original camera terminated unsuccessfully')
        if len(observed) != terminal['frames']:
            raise ValueError('All original published frames require host-time hashes')
        rows = []
        for index in range(terminal['frames']):
            path = directory/f'frame-{index:04d}.json'
            raw = path.read_bytes()
            frame = json.loads(raw)
            diag = {'clock_pairs_present': all(v.get('diagnostic_clock_pair_before') and v.get('diagnostic_clock_pair_after') for v in frame['views'])}
            if (frame.get('request_id') != request_id or frame['index'] != index
                    or frame.get('training_admissible') is not False
                    or frame.get('actor_stream_created') is not False
                    or not diag['clock_pairs_present']
                    or frame.get('generation') != before['generation']):
                raise ValueError('Original diagnostic identity or isolation differs')
            if (sha256(path) != observed[index]['metadata_sha256']
                    or any(sha256(directory/v['file']) != v['sha256']
                           for v in observed[index]['images'].values())):
                raise ValueError('Published frame or image changed since host hash')
            rows.append(dict(index=index, source_file=str(path), sha256=sha256(path),
                             diagnostic=diag))
        if not rows:
            raise ValueError('No original diagnostic observations')
        sampler.stop()
        if sampler.error or len(sampler.rows)<2:raise ValueError('complete native copies required')
        write_json(run/'native-copy-windows.json', sampler.rows)
        for _ in range(4):clock_ping(camera_root, clock, windows[0].hwnd, host_clock=time.perf_counter)
        write_json(run/'clock-after.json', clock.record())
        result.update(native_copy_samples=len(sampler.rows),native_copy_sha256=sha256(run/'native-copy-windows.json'),native_copy_error=sampler.error,clock_after_sha256=sha256(run/'clock-after.json'))
        write_json(run/'diagnostic-frames.json', rows)
        write_json(run/'camera-terminal.json', terminal)
        hash_log.flush()
        result.update(status='completed', frames=len(rows), source_retained=True,
                      camera_terminal=terminal['status'],
                      host_observed_hashes_sha256=sha256(run/'live-frame-hashes.jsonl'),
                      host_hashes_observed_after_publication_not_exposure=True,
                      metadata_sha256=sha256(run/'diagnostic-frames.json'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if child is not None and child.poll() is None:
            result['control_still_live_pid'] = child.pid
        elif directory.exists() and not (directory/'result.json').exists():
            (directory/'stop').touch(exist_ok=True)
        if sampler is not None:
            sampler.stop()
            if not (run/'native-copy-windows.json').exists():
                write_json(run/'native-copy-windows.json', sampler.rows)
                result.update(native_copy_samples=len(sampler.rows),native_copy_sha256=sha256(run/'native-copy-windows.json'),native_copy_error=sampler.error)
        if log:
            log.close()
        if hash_log:
            hash_log.close()
        write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result.get('reason', 'diagnostic failed'))


if __name__ == '__main__':
    main()
