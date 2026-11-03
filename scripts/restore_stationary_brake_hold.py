"""Restore only centered/zero-throttle/full-brake on the reviewed virtual device."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--engine-pid', type=int, required=True)
    p.add_argument('--generation', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding='utf8'))
    run = begin_run(a.output, {'arguments': vars(a), 'program_sha256': sha256(__file__)})
    result = {'status': 'failed', 'actuator_messages_sent': 0, 'ui_actions_sent': 0}
    session = None
    try:
        api = WindowsACWindow()
        windows = api.discover()
        status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        refuse_leased_transport('restore stationary brake hold', status_path=status_path)
        state, modified, _ = read_session_heartbeat(status_path)
        validate_heartbeat(state, time.time()-modified, mutating=True)
        controllers = inspect_vjoy_controllers()
        devices = controllers['vjoy_devices']
        sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
        if (len(windows) != 1 or windows[0].pid != a.engine_pid
                or state['generation'] != a.generation
                or int(state['window_handle'].removesuffix('ULL')) != windows[0].hwnd
                or state.get('is_online') is not False or state.get('ai_controlled') is not False
                or state.get('camera_active') is not False
                or state.get('car_id') != config['car_model']
                or state.get('track_id') != config['track']
                or controllers['candidate_ac_joy_index'] != 0 or len(devices) != 1
                or devices[0]['instance_guid'] != config['instance_guid']
                or sha256(config['profile_path']) != config['profile_sha256']
                or sdk.GetVJDStatus(config['device_id']) != 1):
            raise ValueError('Same offline engine and reviewed free virtual device required')
        with NativeTeacherPhysicsReader() as reader:
            before = reader.receive()
            f = before.physics.fields
            if before.builtin_ai_controlled or abs(f['speedKmh']) >= .1 or f['gas'] >= .02:
                raise ValueError('Native AI off and stopped with zero throttle required')
            write_json(run/'native-before.json', before.authority_record())
            session = VJoySession(sdk, config['device_id'],
                                  [AxisBinding(**b) for b in config['bindings']])
            request = session.open()
            write_json(run/'brake-request.json', request)
            deadline = time.monotonic()+2
            while True:
                sample = reader.receive()
                f = sample.physics.fields
                if sample.builtin_ai_controlled or abs(f['speedKmh']) >= .1 or f['gas'] >= .02:
                    raise ValueError('Stationary AI-off condition changed')
                if f['brake'] > .98:
                    write_json(run/'native-brake-held.json', sample.authority_record())
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Full brake not observed')
                time.sleep(.02)
        result['status'] = 'brake_hold_observed'
    finally:
        if session is not None:
            session.close()
            result['actuator_messages_sent'] = session.submit_attempts
            result['device_free'] = sdk.GetVJDStatus(config['device_id']) == 1
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
