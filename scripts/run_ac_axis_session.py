"""Bounded neutral hold while the game reloads, then a file-triggered stationary probe."""

import argparse
import json
import time
from pathlib import Path

from flybrain_drive.ac_axis_probe import run_stationary_probe
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ac-axis-probe.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    run = begin_run(args.output, dict(configuration=config, scope=__doc__,
                                    startup_deadline_seconds=300))
    session = None
    try:
        controllers = inspect_vjoy_controllers()
        devices = controllers['vjoy_devices']
        if (controllers['candidate_ac_joy_index'] != 0 or len(devices) != 1
                or devices[0]['instance_guid'] != config['instance_guid']):
            raise ValueError('Controller identity/index changed')
        sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
        session = VJoySession(sdk, config['device_id'],
                              [AxisBinding(**b) for b in config['bindings']])
        first = session.open()
        write_json(run / 'ready.json', dict(status='neutral_hold', first_request=first,
            ranges=session.ranges, game_applied=False))
        deadline = time.monotonic() + 300
        with (run / 'requests-and-physics.jsonl').open('x', encoding='utf8') as stream:
            def emit(event):
                event = dict(event)
                for key in ('physics_before', 'physics_after'):
                    if key in event:
                        event[key] = physics_record(event[key])
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
            emit(dict(kind='initial_hold', request=first))
            while not (run / 'begin-probe').exists():
                if (run / 'stop').exists():
                    raise RuntimeError('Explicit stop file received')
                if time.monotonic() >= deadline:
                    raise TimeoutError('No probe trigger within startup budget')
                emit(dict(kind='neutral_hold', request=session.submit(0., 0., 1.)))
                time.sleep(.05)
            native = WindowsACWindow()
            windows = native.discover()
            if len(windows) != 1:
                raise ValueError('Exactly one original AC engine required')
            identity = windows[0]

            def guard():
                if (run / 'stop').exists():
                    raise RuntimeError('Explicit stop file received')
                if sha256(config['profile_path']) != config['profile_sha256']:
                    raise RuntimeError('Reviewed profile differs')
                if native.probe(identity.hwnd, require_foreground=False) != identity:
                    raise RuntimeError('Game window changed')
            with NativePhysicsReader(timeout=.25) as reader:
                result = run_stationary_probe(session, reader.receive, guard, emit,
                    **config['budgets'], already_open=True)
            result.update(actuator_messages_sent=session.submit_attempts,
                          axis_response_verified=False, goal_achieved=False)
            write_json(run / 'result.json', result)
    except BaseException as error:
        write_json(run / 'result.json', dict(status='failed', error=type(error).__name__,
            message=str(error), actuator_messages_sent=session.submit_attempts if session else 0,
            goal_achieved=False))
        raise
    finally:
        if session is not None and session.owned:
            try:
                receipt = session.close()
                write_json(run / 'final-neutral-request.json', receipt)
            except Exception as error:
                write_json(run / 'cleanup-error.json', dict(message=str(error)))


if __name__ == '__main__':
    main()
