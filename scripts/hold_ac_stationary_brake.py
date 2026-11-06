"""Establish only centered full brake after engine startup; no throttle or steering task."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import validate_heartbeat
from scripts.record_ac_wgc import physics_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config', type=Path, default=Path('configs/ac-axis-probe.json'))
    p.add_argument('--output', required=True)
    a = p.parse_args()
    axis = json.loads(a.axis_config.read_text(encoding='utf8'))
    run = begin_run(a.output, dict(scope=__doc__, axis=axis, bounded_requests=27))
    result = dict(status='failed', goal_achieved=False, ui_actions_sent=0)
    session = reader = ai = None
    requests, samples = [], []
    try:
        if axis["car_model"] != "bmw_z4_gt3" or axis["track"] != "monza":
            raise ValueError("Reviewed Monza BMW Z4 GT3 axis configuration required")
        api = WindowsACWindow()
        windows = api.discover()
        assert len(windows) == 1
        window = windows[0]
        controllers = inspect_vjoy_controllers()
        assert controllers['candidate_ac_joy_index'] == 0
        assert len(controllers['vjoy_devices']) == 1
        assert controllers['vjoy_devices'][0]['instance_guid'] == axis['instance_guid']
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        reader, ai = NativePhysicsReader(timeout=.15), NativeTeacherStatusReader()

        def checked_sample():
            state, modified = read_published_json(
                Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
            validate_heartbeat(state, time.time()-modified, mutating=True)
            if (state['is_online'] or state['camera_active'] or state['ai_controlled'] is not False
                    or state['car_id'] != 'bmw_z4_gt3' or state['track_id'] != 'monza'
                    or int(state['window_handle'].removesuffix('ULL')) != window.hwnd
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or api.user.GetForegroundWindow() == window.hwnd
                    or sha256(axis['profile_path']) != axis['profile_sha256']
                    or ai.snapshot()['builtin_ai_controlled']):
                raise ValueError('Same idle reviewed background AI-off engine required')
            sample = reader.receive()
            samples.append(physics_record(sample))
            if abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02:
                raise ValueError(
                    'Only already stationary zero-throttle brake initialization allowed')
            return sample

        checked_sample()
        session = VJoySession(sdk, axis['device_id'], [AxisBinding(**b) for b in axis['bindings']])
        requests.append(session.open())
        settled = 0
        for _ in range(25):
            checked_sample()
            requests.append(session.submit(0., 0., 1.))
            time.sleep(.04)
            sample = checked_sample()
            settled = settled+1 if sample.fields['brake'] > .98 else 0
            if settled >= 5:
                break
        else:
            raise ValueError(
                'Native full-brake response did not settle within finite request budget')
        requests.append(session.close())
        post = checked_sample()
        assert post.fields['brake'] > .98 and abs(post.fields['steerAngle']) < .005
        assert sdk.GetVJDStatus(axis['device_id']) == 1
        result.update(status='completed_stationary_brake_hold', released=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if session is not None and session.owned:
            try:
                requests.append(session.close())
            except Exception as error:
                result['cleanup_error'] = str(error)
        if reader is not None:
            reader.close()
        if ai is not None:
            ai.close()
        write_json(run/'requests.json', requests)
        write_json(run/'samples.json', samples)
        result['actuator_messages_sent'] = session.submit_attempts if session else 0
        write_json(run/'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
