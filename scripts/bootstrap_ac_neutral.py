"""Native-only stationary zero-throttle/full-brake bootstrap; no steering or SDK substitute."""
import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    axis = json.loads(Path(a.axis_config).read_text(encoding='utf8'))
    run = begin_run(a.output, dict(arguments=vars(a), axis_configuration=axis,
        seed=780102, only_request=[0, 0, 1], steering_changes=0, throttle_requests=0))
    session = None
    result = dict(status='failed', ui_actions_sent=0, model_action_requests=0)
    reader, flag = NativePhysicsReader(timeout=.15), NativeTeacherStatusReader()
    try:
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one engine required')
        window = windows[0]
        def read():
            hb, modified = read_published_json(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
            sample, ai = reader.receive(), flag.snapshot()
            if (not 0 <= time.time()-modified <= 2 or hb['paused'] or hb['in_menu'] or hb['is_online']
                    or hb['ai_controlled'] or hb['camera_active'] or hb['expert_probe_active']
                    or hb.get('error') or hb.get('pending_receipt_id') or ai['builtin_ai_controlled']
                    or hb['car_id'] != axis['car_model'] or hb['track_id'] != axis['track']
                    or int(hb['window_handle'].removesuffix('ULL')) != window.hwnd
                    or api.user.GetForegroundWindow() == window.hwnd
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] > .02
                    or sha256(axis['profile_path']) != axis['profile_sha256']):
                raise ValueError('Fresh stationary live AI-off original profile required')
            return hb, sample, ai
        hb, sample, ai = read()
        generation = hb['generation']
        write_json(run/'before.json', dict(session=hb, native=physics_record(sample), ai=ai))
        if not a.execute:
            result['status'] = 'read_only'
        else:
            sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
            session = VJoySession(sdk, axis['device_id'], [AxisBinding(**b) for b in axis['bindings']])
            write_json(run/'request.json', session.open())
            deadline = time.perf_counter()+2
            while True:
                hb, sample, ai = read()
                if hb['generation'] != generation:
                    raise ValueError('Generation changed')
                if sample.fields['brake'] >= .98:
                    write_json(run/'acknowledged.json', dict(session=hb, native=physics_record(sample), ai=ai))
                    result['status'] = 'completed'
                    break
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Full brake was not acknowledged')
                time.sleep(.01)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if session and session.owned:
            write_json(run/'close.json', session.close())
        reader.close()
        flag.close()
        result['actuator_messages_sent'] = session.submit_attempts if session else 0
        write_json(run/'result.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
