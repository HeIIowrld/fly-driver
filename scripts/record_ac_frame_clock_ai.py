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
                      '/FlyBrainCamera-frame-clock-diagnostic-v1.lua')
    if sha256(deployed) != sha256(candidate):
        raise ValueError('Installed candidate differs')
    request_id = 'airflow-' + uuid.uuid4().hex
    request = dict(id=request_id, frames=250, period_seconds=.04, deadline_seconds=15,
                   exposure=.35, image_format='png', airflow_diagnostic=True,
                   deferred_save_probe=True,
                   expected_generation=before['generation'])
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), request=request,
        candidate_sha256=sha256(candidate), before=before, actor_deployed=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', request_id=request_id, actor_deployed=False,
                  ui_actions_sent=0, student_actuator_messages_sent=0)
    child, log, hash_log = None, None, None
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
                    or frame.get('deferred_save_probe') is not True
                    or frame.get('aid_probe_only') is not True
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
        if (frame['request_id'] != request_id or not frame.get('aid_probe_only')
                or not frame.get('airflow_diagnostic')):
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
        child = subprocess.Popen([sys.executable, '-m', 'scripts.control_ac_frame_clock_ai',
            '--command', 'expert-motion-probe', '--execute', '--camera-request-id', request_id,
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
            diag = frame.get('airflow_diagnostic', {})
            if (frame.get('request_id') != request_id or frame['index'] != index
                    or not frame.get('aid_probe_only')
                    or frame.get('deferred_save_probe') is not True
                    or diag.get('source_verified') is not False
                    or diag.get('actor_admissible') is not False
                    or diag.get('generation') != before['generation']):
                raise ValueError('Original diagnostic identity or isolation differs')
            if (sha256(path) != observed[index]['metadata_sha256']
                    or any(sha256(directory/v['file']) != v['sha256']
                           for v in observed[index]['images'].values())):
                raise ValueError('Published frame or image changed since host hash')
            rows.append(dict(index=index, source_file=str(path), sha256=sha256(path),
                             diagnostic=diag))
        if not rows:
            raise ValueError('No original diagnostic observations')
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
        if log:
            log.close()
        if hash_log:
            hash_log.close()
        write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result.get('reason', 'diagnostic failed'))


if __name__ == '__main__':
    main()
