"""Read-only preflight by default; --execute runs the bounded stationary axis experiment."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from flybrain_drive.ac_axis_probe import run_stationary_probe
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_static_identity_prefix
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ac-axis-probe.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    run = begin_run(args.output, dict(arguments=vars(args), configuration=config, scope=__doc__))
    requests, session = 0, None
    try:
        native = WindowsACWindow()
        windows = native.discover()
        controllers = inspect_vjoy_controllers()
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
        actual_profile_sha = sha256(config['profile_path'])
        reasons = []
        if actual_profile_sha != config['profile_sha256']:
            reasons.append('active controls.ini differs from the reviewed vJoy candidate')
        if len(windows) != 1:
            reasons.append('exactly one original AC engine window required')
        devices = controllers['vjoy_devices']
        if (controllers['candidate_ac_joy_index'] != 0 or len(devices) != 1
                or devices[0]['instance_guid'] != config['instance_guid']):
            reasons.append('DirectInput identity/index differs from candidate profile')
        if identity['car_model'] != config['car_model'] or identity['track'] != config['track']:
            reasons.append('car/track differs from stationary experiment configuration')
        if not sdk.vJoyEnabled() or sdk.GetVJDStatus(config['device_id']) != 1:
            reasons.append('vJoy must be enabled and device FREE')
        write_json(run / 'preflight.json', dict(
            profile_sha256=actual_profile_sha, expected_profile_sha256=config['profile_sha256'],
            windows=[asdict(w) for w in windows], controllers=controllers, identity=identity,
            reasons=reasons, ready=not reasons, device_acquisitions=0, actuator_messages_sent=0,
            profile_reload_verified=False))
        if not args.execute:
            write_json(run / 'result.json', dict(status='preflight_only', ready=not reasons,
                reasons=reasons, device_acquisitions=0, actuator_messages_sent=0,
                axis_response_verified=False, goal_achieved=False))
            return
        if reasons:
            raise RuntimeError('; '.join(reasons))
        hwnd, initial = windows[0].hwnd, windows[0]

        def guard():
            if sha256(config['profile_path']) != config['profile_sha256']:
                raise RuntimeError('Control profile changed during experiment')
            if native.probe(hwnd, require_foreground=False) != initial:
                raise RuntimeError('Game window producer/geometry changed during experiment')

        session = VJoySession(sdk, config['device_id'],
                              [AxisBinding(**b) for b in config['bindings']])
        with NativePhysicsReader(timeout=.25, poll_seconds=.005) as reader:
            with (run / 'requests-and-physics.jsonl').open('x', encoding='utf8') as stream:
                def emit(event):
                    nonlocal requests
                    if event.get('request', {}).get('driver_accepted'):
                        requests += 1
                    event = dict(event)
                    event['was_foreground_at_record'] = bool(
                        native.user.GetForegroundWindow() == hwnd)
                    for key in ('physics_before', 'physics_after'):
                        if key in event:
                            event[key] = physics_record(event[key])
                    stream.write(json.dumps(event, allow_nan=False) + '\n')
                    stream.flush()
                result = run_stationary_probe(session, reader.receive, guard, emit,
                                               **config['budgets'])
        result.update(actuator_messages_sent=session.submit_attempts,
                      driver_accepted_requests=requests, axis_response_verified=False,
                      game_profile_reload_verified=False, goal_achieved=False)
        write_json(run / 'result.json', result)
    except BaseException as error:
        write_json(run / 'result.json', dict(status='failed', error=type(error).__name__,
            message=str(error), actuator_messages_sent=session.submit_attempts if session else 0,
            driver_accepted_requests=requests, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
