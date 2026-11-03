"""Deploy a pinned publisher and request one CSP engine restart; no UI or actuator input."""

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.control_ac_session import validate_heartbeat
from scripts.record_ac_wgc import physics_record


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-installed-publisher-sha256', required=True)
    p.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    p.add_argument('--expected-lua-sha256')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source = Path('integrations/ac/flybrain_telemetry/flybrain_telemetry.py')
    run = begin_run(a.output, dict(arguments=vars(a), scope=__doc__,
                                  new_publisher_sha256=sha256(source)))
    result = dict(status='failed', goal_achieved=False, ui_actions_sent=0,
                  actuator_messages_sent=0, lifecycle_requests_sent=0, publisher_installed=False)
    old_process = None
    root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    telemetry = Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry'
    try:
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original engine required')
        window = windows[0]
        axis = read(a.axis_config)
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        deployed_lua = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        if sha256(deployed_lua) != (a.expected_lua_sha256 or sha256(
                'integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')):
            raise ValueError('Same reviewed deployed lifecycle source required')

        def stationary_preflight():
            state, modified = read_published_json(root/'status.json')
            validate_heartbeat(state, time.time()-modified, mutating=True)
            if (state['paused'] is not True or state['is_online']
                    or state['camera_active'] or state['ai_controlled'] is not False
                    or state['car_id'] != 'bmw_z4_gt3' or state['track_id'] != 'monza'
                    or int(state['window_handle'].removesuffix('ULL')) != window.hwnd
                    or state['capabilities'].get('restart_engine') is not True
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or api.user.GetForegroundWindow() == window.hwnd
                    or sdk.GetVJDStatus(axis['device_id']) != 1
                    or sha256(axis['profile_path']) != axis['profile_sha256']):
                raise ValueError('Reviewed paused background AI-off idle engine required')
            with NativePhysicsReader(timeout=.15) as reader:
                sample = reader.receive()
            teacher = NativeTeacherStatusReader()
            try:
                ai = teacher.snapshot()
            finally:
                teacher.close()
            if (abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
                    or sample.fields['brake'] <= .98 or ai['builtin_ai_controlled']):
                raise ValueError('Independent stationary native brake hold and AI-off required')
            return state, sample

        before, sample = stationary_preflight()
        old_process = ProcessReceipt(window.pid)
        assert old_process.snapshot()['active']
        old_producer = FileTelemetryClient(telemetry, timeout=.15, clock=time.perf_counter)
        write_json(run/'before.json', dict(session=before, window=asdict(window),
            native=physics_record(sample), producer_session=old_producer.session_id))
        target = (Path(window.executable).parent
                  / 'apps/python/flybrain_telemetry/flybrain_telemetry.py')
        if sha256(target) != a.expected_installed_publisher_sha256:
            raise ValueError('Publisher installation changed; never overwrite an unreviewed file')
        if not a.execute:
            result.update(status='read_only_preflight', old_engine_pid=window.pid)
        else:
            backup = run/'publisher-installed-before.py'
            backup.write_bytes(target.read_bytes())
            assert sha256(backup) == a.expected_installed_publisher_sha256
            staged = target.with_name(target.name+'.flybrain-partial')
            if staged.exists():
                raise ValueError('Existing publisher deployment partial requires inspection')
            staged.write_bytes(source.read_bytes())
            expected = read(run/'manifest.json')['config']['new_publisher_sha256']
            assert sha256(staged) == sha256(source) == expected
            fresh, _ = stationary_preflight()
            assert fresh['generation'] == before['generation'] and fresh['epoch'] == before['epoch']
            os.replace(staged, target)
            result['publisher_installed'] = True
            assert sha256(target) == expected
            request = dict(schema='flybrain.csp.session.request.v1', id=uuid.uuid4().hex,
                command='restart-engine', generation=fresh['generation'], epoch=fresh['epoch'],
                expires_clock=fresh['script_clock']+3, old_engine_pid=window.pid,
                released_controller_verified=True)
            write_json(run/'request.json', request)
            write_json(root/'request.json', request)
            result.update(lifecycle_requests_sent=1, old_engine_pid=window.pid,
                          request_id=request['id'], restart_outcome_verified=False)
            deadline, next_progress = time.monotonic()+90, 0.
            while time.monotonic() < deadline:
                old = old_process.snapshot()
                intent_path = root/(request['id']+'.terminal.json')
                if intent_path.exists() and not (run/'restart-intent.json').exists():
                    intent = read(intent_path)
                    assert intent['id'] == request['id']
                    assert intent['status'] == 'engine_restart_intent_committed'
                    write_json(run/'restart-intent.json', intent)
                candidates = api.discover()
                if time.monotonic() >= next_progress:
                    write_json(run/'progress.json', dict(original_process=old,
                        observed_windows=[asdict(w) for w in candidates], request_id=request['id']))
                    next_progress = time.monotonic()+1
                newer = [w for w in candidates if w.pid != window.pid
                         and Path(w.executable) == Path(window.executable)]
                if not old['active'] and len(newer) == 1:
                    new_window = newer[0]
                    try:
                        state, modified = read_published_json(root/'status.json')
                        validate_heartbeat(state, time.time()-modified, mutating=False)
                        client = FileTelemetryClient(telemetry, timeout=.02,
                                                     clock=time.perf_counter)
                        packet = client.snapshot()
                    except (OSError, ValueError, TimeoutError):
                        time.sleep(.1)
                        continue  # Same restart request and original process handle, never retry.
                    if (state['generation'] != before['generation']
                            and int(state['window_handle'].removesuffix('ULL')) == new_window.hwnd
                            and client.session_id != old_producer.session_id
                            and client.session_id.startswith('ac-file-'+str(new_window.pid)+'-')
                            and packet.raw.get('publisher_replace_retry_limit_seconds') == .003
                            and isinstance(packet.raw.get('publisher_replace_retries'), int)):
                        assert state['car_id'] == 'bmw_z4_gt3' and state['track_id'] == 'monza'
                        assert not state['is_online'] and not state['camera_active']
                        assert state['ai_controlled'] is False and sha256(target) == expected
                        write_json(run/'after.json', dict(session=state,
                            window=asdict(new_window), producer_session=client.session_id,
                            packet=record_packet(packet), original_process=old))
                        result.update(status='completed_engine_publisher_restart',
                            restart_outcome_verified=True, new_engine_pid=new_window.pid,
                            new_producer_session=client.session_id,
                            foreground_after=bool(
                                api.user.GetForegroundWindow() == new_window.hwnd),
                            new_engine_paused=state['paused'], new_engine_in_menu=state['in_menu'])
                        break
                time.sleep(.1)
            else:
                raise TimeoutError(
                    'Same engine restart outcome unknown; inspect original handle/PIDs')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if old_process is not None:
            result['original_process'] = old_process.snapshot()
            old_process.close()
        write_json(run/'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
