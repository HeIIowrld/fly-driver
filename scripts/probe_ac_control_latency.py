"""Stationary background input/observation timing; no throttle, focus or student policy."""

import argparse
import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_physics_prefix
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_pilot import validate_calibration
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def summarize_steps(commands, samples, *, native_steering_gain=1.4062499006589255,
                    sdk_steering_gain=449.9999491373698):
    """First observed target response, not physical actuator application or exposure time."""
    result = []
    for command in commands:
        row = dict(index=command['index'], axis=command['axis'], target=command['target'])
        for source, scale in [('native', native_steering_gain), ('sdk', sdk_steering_gain)]:
            selected = [s for s in samples if s['source'] == source
                        and command['after_perf'] <= s['after_perf'] <= command['end_perf']]
            key = 'steerAngle' if command['axis'] == 'steering' else \
                ('brake' if source == 'native' else 'brakeStatus')
            target = command['target'] * scale \
                if command['axis'] == 'steering' else command['target']
            values = [s['values'][key] for s in selected]
            tail = [s['values'][key] for s in selected
                    if s['after_perf'] >= command['end_perf']-.06]
            tolerance = .001 if source == 'native' and command['axis'] == 'steering' else \
                .1 if command['axis'] == 'steering' else .002
            hit = next((s for s in selected if abs(s['values'][key]-target) <= tolerance), None)
            preceding = [s for s in samples if s['source'] == source
                         and s['after_perf'] <= command['before_perf']]
            baseline = max(preceding, key=lambda s: s['after_perf'])['values'][key] \
                if preceding else None
            transition = baseline is not None and abs(baseline-target) > tolerance
            row[source] = dict(samples=len(values), tail_samples=len(tail),
                steady_median=float(np.median(tail)) if tail else None, expected=target,
                previous_observed_value=baseline, target_differs_from_baseline=transition,
                first_target_observed_ms=(hit['after_perf']-command['before_perf'])*1000
                    if hit and transition else None, match_tolerance=tolerance)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    parser.add_argument('--calibration-result', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    axis = json.loads(Path(args.axis_config).read_text(encoding='utf8'))
    calibration = json.loads(Path(args.calibration_result).read_text(encoding='utf8'))
    # Native gain is a measured game-signal scale, never a roadwheel-radian conversion.
    samples_path = Path(args.calibration_result).parent / 'samples.jsonl'
    calibration_rows = [json.loads(line) for line in
                        samples_path.read_text(encoding='utf8').splitlines()]
    stages = [[] for _ in range(4)]
    for row in calibration_rows:
        if 'native' in row:
            raw = base64.b64decode(row['native']['prefix_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != row['native']['prefix_sha256']:
                raise ValueError('Independent native calibration bytes changed')
            sample = decode_physics_prefix(raw, row['native']['received_monotonic_seconds'])
            stages[row['stage']].append(sample.fields['steerAngle'])
    if any(len(stage) != 8 for stage in stages):
        raise ValueError('32 independent stationary native calibration samples required')
    medians = np.array([np.median(stage) for stage in stages])
    native_gain = float((medians[1]-medians[2])/(2*calibration['request_amplitude']))
    if not native_gain > 0 or max(abs(medians[[0, 3]])) > .001:
        raise ValueError('Independent native steering calibration differs')
    run = begin_run(args.output, dict(arguments=vars(args), axis_configuration=axis,
        scope=__doc__, seed=861501, speed_limit_kmh=.5, displacement_limit_m=.1,
        throttle=0., perf_counter_scope='collector-host observations and API-call brackets only',
        native_steering_gain=native_gain, calibration_samples_sha256=sha256(samples_path),
        physical_actuator_application_time_verified=False))
    stop, lock = threading.Event(), threading.Lock()
    samples, errors, commands, threads = [], [], [], []
    reader, status_reader, session, sdk = None, None, None, None
    result = dict(status='not_ready', actuator_messages_sent=0, ui_actions_sent=0,
                  throttle=0., goal_achieved=False)
    try:
        windows_api = WindowsACWindow()
        windows = windows_api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original game engine window required')
        window = windows[0]
        if sha256(axis['profile_path']) != axis['profile_sha256']:
            raise ValueError('Active control profile changed')
        controllers = inspect_vjoy_controllers()
        if (controllers['candidate_ac_joy_index'] != 0
                or len(controllers['vjoy_devices']) != 1
                or controllers['vjoy_devices'][0]['instance_guid'] != axis['instance_guid']):
            raise ValueError('Reviewed DirectInput identity/index required')
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                      timeout=.15, poll_seconds=.001)
        reader = NativePhysicsReader(timeout=.15, poll_seconds=.001)
        status_reader = NativeTeacherStatusReader()
        first, packet = reader.receive(), client.receive()
        if abs(first.fields['speedKmh']) > .1:
            raise ValueError('Stationary car required')
        truth = packet.raw['privileged_teacher']['raw']
        origin = np.array([truth['world_position_x'], truth['world_position_y']])
        validate_calibration(calibration, axis, client.session_id, steering_request_scale=.6)
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        if sdk.GetVJDStatus(axis['device_id']) != 1:
            raise ValueError('vJoy must be FREE')
        if status_reader.snapshot()['builtin_ai_controlled']:
            raise ValueError('Builtin AI owns controls')
        if not args.execute:
            result.update(status='preflight_only')
        else:
            def collect(source, receive):
                try:
                    while not stop.is_set():
                        before = time.perf_counter()
                        item = receive()
                        after = time.perf_counter()
                        if source == 'native':
                            values = dict(item.fields)
                            raw = physics_record(item)
                        else:
                            values = {k: item.raw[k] for k in
                                      ('steerAngle', 'accStatus', 'brakeStatus', 'speed')}
                            p = item.raw['privileged_teacher']['raw']
                            values.update(x=p['world_position_x'], y=p['world_position_y'])
                            raw = record_packet(item)
                        with lock:
                            if len(samples) >= 4000:
                                raise RuntimeError('Finite sensor sample budget exhausted')
                            samples.append(dict(source=source, before_perf=before,
                                after_perf=after, values=values, raw=raw))
                except Exception as error:
                    if not stop.is_set():
                        errors.append(str(error))

            for source, receive in [('native', reader.receive), ('sdk', client.receive)]:
                thread = threading.Thread(target=collect, args=(source, receive), daemon=True)
                threads.append(thread)
                thread.start()
            time.sleep(.05)

            def guard():
                if errors or (run / 'stop').exists():
                    raise RuntimeError(str(errors) if errors else 'Owned stop file received')
                if (windows_api.probe(window.hwnd, require_foreground=False) != window
                        or sha256(axis['profile_path']) != axis['profile_sha256']
                        or status_reader.snapshot()['builtin_ai_controlled']):
                    raise ValueError('Engine/control identity or ownership changed')
                now = time.perf_counter()
                with lock:
                    native = next((r for r in reversed(samples) if r['source'] == 'native'), None)
                    sdk_row = next((r for r in reversed(samples) if r['source'] == 'sdk'), None)
                if (native is None or sdk_row is None or now-native['after_perf'] > .05
                        or now-sdk_row['after_perf'] > .15):
                    raise ValueError('Fresh independent native/SDK observations required')
                if abs(native['values']['speedKmh']) > .5:
                    raise ValueError('Stationary timing probe speed limit exceeded')
                xy = np.array([sdk_row['values']['x'], sdk_row['values']['y']])
                if np.linalg.norm(xy-origin) > .1:
                    raise ValueError('Stationary timing probe displacement exceeded')

            guard()
            session = VJoySession(sdk, axis['device_id'],
                                  [AxisBinding(**b) for b in axis['bindings']])
            result['open'] = session.open()
            time.sleep(.1)
            program = [('steering', v, .2) for v in (0., .05, -.05, 0.)]
            program += [('brake', v, .15) for v in
                        (1., .1, .05, .03, .025, .02, .015, .01, .005, 0., .03, 1.)]
            for index, (axis_name, value, duration) in enumerate(program):
                guard()
                before = time.perf_counter()
                request = session.submit(value if axis_name == 'steering' else 0., 0.,
                                         1. if axis_name == 'steering' else value)
                after = time.perf_counter()
                command = dict(index=index, axis=axis_name, target=value, request=request,
                    before_perf=before, after_perf=after, end_perf=after+duration,
                    was_foreground=bool(windows_api.user.GetForegroundWindow() == window.hwnd))
                commands.append(command)
                while time.perf_counter() < command['end_perf']:
                    guard()
                    stop.wait(.005)
            result.update(status='completed')
    except Exception as error:
        result.update(status='aborted', error=type(error).__name__, reason=str(error))
    except BaseException as error:
        result.update(status='interrupted', error=type(error).__name__, reason=str(error))
        raise
    finally:
        if session is not None:
            try:
                if session.owned:
                    result['close'] = session.close()
                time.sleep(.15)
            except Exception as error:
                result['close_error'] = str(error)
            result['actuator_messages_sent'] = session.submit_attempts
        stop.set()
        for thread in threads:
            thread.join(timeout=.5)
        result['collector_threads_alive'] = any(thread.is_alive() for thread in threads)
        if reader is not None:
            reader.close()
        if status_reader is not None:
            status_reader.close()
        if sdk is not None:
            result['final_vjoy_status'] = sdk.GetVJDStatus(axis['device_id'])
        write_json(run / 'commands.json', commands)
        with (run / 'sensor-events.jsonl').open('x', encoding='utf8') as stream:
            for row in sorted(samples, key=lambda row: row['after_perf']):
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        result['steps'] = summarize_steps(commands, samples, native_steering_gain=native_gain,
            sdk_steering_gain=calibration['sdk_raw_per_normalized_request'])
        result['collector_errors'] = errors
        result['sensor_samples'] = len(samples)
        result['foreground_commands'] = sum(r['was_foreground'] for r in commands)
        result['physical_application_time_verified'] = False
        result['exposure_sync_verified'] = False
        write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
