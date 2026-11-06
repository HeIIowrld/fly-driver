"""Finite neutral-to-first-gear diagnostic, not teacher or fly policy driving."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, NativePhysicsReader
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_pilot import validate_calibration
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-before-first-gear',action='store_true')
    parser.add_argument('--thermal-initialization-run')
    parser.add_argument('--pause-after-first-gear',action='store_true',help='One owned API pause immediately after observed stationary first gear')
    parser.add_argument('--expected-lua-sha256')
    parser.add_argument('--ensure-first-gear',action='store_true',
                        help='Accept observed existing first gear without another button press')
    parser.add_argument('--telemetry-transport', choices=('file', 'tcp'), default='file',
                        help='TCP requires original source-native timestamps and a gear-only request')
    parser.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    parser.add_argument('--gear-up-button', type=int,
                        help='Explicit zero-based button; gear-only test with no throttle pulse')
    parser.add_argument('--calibration-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.resume_before_first_gear and (not args.execute or not args.pause_after_first_gear or not args.thermal_initialization_run):
        raise ValueError('Owned gear resume requires explicit thermal receipt, gear-only execution and final pause')
    if args.pause_after_first_gear and (not args.execute or args.gear_up_button is None or not args.expected_lua_sha256):
        raise ValueError('First-gear freeze requires explicit execution, reviewed gear-only button and lifecycle source')
    config = json.loads(Path(args.axis_config).read_text(encoding='utf8'))
    if (args.gear_up_button is not None
            and config.get('gear_buttons_zero_based', {}).get('up') != args.gear_up_button):
        raise ValueError('Button must match explicitly reviewed game profile configuration')
    if args.telemetry_transport == 'tcp' and args.gear_up_button is None:
        raise ValueError('TCP is restricted to the stationary gear-only probe')
    calibration = json.loads(Path(args.calibration_result).read_text())
    run = begin_run(args.output, dict(arguments=vars(args), axis_configuration=config,
        calibration=calibration, scope=__doc__,
        throttle=0 if args.gear_up_button is not None else .05,
        pulse_seconds=0 if args.gear_up_button is not None else .4,
        speed_limit_kmh=2., displacement_limit_m=.25, ui_actions_sent=0))
    reader, status, session, client = None, None, None, None
    events = []
    result = dict(status='not_ready', ui_actions_sent=0, goal_achieved=False)
    try:
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('One original AC engine required')
        window = windows[0]
        if args.telemetry_transport == 'tcp':
            from flybrain_drive.ac_latest_tcp import LatestTcpTelemetryClient
            from flybrain_drive.source_native_snapshot import WindowsQpc, source_age_seconds
            clock = WindowsQpc()
            client = LatestTcpTelemetryClient(
                source_admission=lambda packet: source_age_seconds(packet, clock) <= .05,
                maximum_packets=500)
        else:
            client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                         timeout=.15)
        reader, status = NativePhysicsReader(timeout=.15), NativeTeacherStatusReader()
        origin = None

        def receive():
            if (sha256(config['profile_path']) != config['profile_sha256']
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or status.snapshot()['builtin_ai_controlled']
                    or args.telemetry_transport == 'tcp' and api.user.GetForegroundWindow() == window.hwnd):
                raise ValueError('Engine/profile/control ownership changed')
            packet, physics = client.receive(), reader.receive()
            namespace = packet.raw['privileged_teacher']
            if namespace['actor_input'] is not False:
                raise ValueError('Privileged diagnostic-only position required')
            extra = namespace['raw']
            position = np.array([extra['world_position_x'], extra['world_position_y']], dtype=float)
            if not np.isfinite(position).all():
                raise ValueError('Position unavailable; no displacement bound')
            values = PHYSICS_PREFIX.unpack(physics.prefix_bytes)
            raw, offset = {}, 0
            for name, _, width in PHYSICS_LAYOUT:
                raw[name] = values[offset] if width == 1 else values[offset:offset + width]
                offset += width
            row = dict(sdk=record_packet(packet), native=physics_record(physics),
                gear=raw['gear'], automatic_shifter=raw['autoShifterOn'],
                speed_kmh=physics.fields['speedKmh'], position=position.tolist(),
                displacement_m=float(np.linalg.norm(position - origin))
                    if origin is not None else 0,
                foreground=bool(api.user.GetForegroundWindow() == window.hwnd))
            events.append(row)
            if raw['gear'] not in (1, 2) or raw['autoShifterOn'] != 1:
                raise ValueError('Neutral/first gear and automatic shifting required')
            return packet, row

        packet, first = receive()
        origin = np.array(first['position'])
        validate_calibration(calibration, config, packet.raw['producer_session_id'],
                             steering_request_scale=.6)
        if abs(first['speed_kmh']) > .1:
            raise ValueError('Car must already be stationary')
        if args.resume_before_first_gear:
            from flybrain_drive.thermal_gear_resume import resume_prepared_thermal_for_gear
            sdk=load_vjoy_sdk(config['dll_path'],config['dll_sha256'])
            resume_prepared_thermal_for_gear(output=run/'owned-thermal-resume',window=window,axis=config,sdk=sdk,source_sha256=args.expected_lua_sha256,initialization=args.thermal_initialization_run)
            packet,first=receive()
        if args.ensure_first_gear and first['gear']==2:
            result.update(status='completed',first_gear_observed=True,
                already_in_first=True,max_speed_kmh=abs(first['speed_kmh']),
                final_speed_kmh=first['speed_kmh'],final_brake=first['native']['fields']['brake'])
            if args.pause_after_first_gear:
                from flybrain_drive.first_gear_freeze import pause_first_gear
                sdk=load_vjoy_sdk(config['dll_path'],config['dll_sha256'])
                pause_first_gear(output=run/'first-gear-freeze',window=window,axis=config,sdk=sdk,reader=reader,source_sha256=args.expected_lua_sha256)
                result['paused_first_gear_verified']=True
            return
        if args.gear_up_button is not None and first['gear'] != 1:
            raise ValueError('Begin in neutral for the single-button test; never overshift')
        if not args.execute:
            result.update(status='preflight_only', ready=True)
        else:
            sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
            session = VJoySession(sdk, config['device_id'],
                                  [AxisBinding(**b) for b in config['bindings']])
            opening = session.open()
            write_json(run / 'opening-request.json', opening)
            for _ in range(4):
                time.sleep(.04)
                _, held = receive()
                if (abs(held['speed_kmh']) > .1 or held['native']['fields']['gas'] > .02
                        or held['native']['fields']['brake'] < .98):
                    raise ValueError('Stationary neutral hold not acknowledged')
            if args.gear_up_button is not None:
                button_receipt = session.submit(0., 0., 1.,
                    pressed_buttons=(args.gear_up_button,))
                write_json(run / 'gear-up-request.json', button_receipt)
                time.sleep(.12)
                write_json(run / 'button-release-request.json', session.submit(0., 0., 1.))
            deadline = time.monotonic() + (0 if args.gear_up_button is not None else .4)
            limit_hit = False
            while time.monotonic() < deadline:
                _, row = receive()
                if abs(row['speed_kmh']) >= 2 or row['displacement_m'] >= .25:
                    limit_hit = True
                    break
                if time.monotonic() >= deadline:
                    break
                row['request'] = session.submit(0., .05, 0.)
                time.sleep(min(.04, max(0., deadline - time.monotonic())))
            braking = session.submit(0., 0., 1.)
            write_json(run / 'braking-request.json', braking)
            for _ in range(20):
                time.sleep(.04)
                _, row = receive()
                if abs(row['speed_kmh']) < .1 and row['native']['fields']['brake'] >= .98:
                    break
            result.update(status='completed', limit_hit=limit_hit,
                first_gear_observed=any(e['gear'] == 2 for e in events),
                max_speed_kmh=max(abs(e['speed_kmh']) for e in events),
                max_displacement_m=max(e['displacement_m'] for e in events),
                final_speed_kmh=row['speed_kmh'], final_brake=row['native']['fields']['brake'])
            if args.pause_after_first_gear:
                if not result['first_gear_observed']:raise ValueError('First gear not observed; cannot freeze')
                write_json(run/'closing-request.json',session.close())
                from flybrain_drive.first_gear_freeze import pause_first_gear
                pause_first_gear(output=run/'first-gear-freeze',window=window,axis=config,sdk=sdk,reader=reader,source_sha256=args.expected_lua_sha256)
                result['paused_first_gear_verified']=True
    except Exception as error:
        result.update(status='aborted', error=type(error).__name__, reason=str(error))
    except BaseException as error:
        result.update(status='interrupted', error=type(error).__name__, reason=str(error))
        raise
    finally:
        if session is not None and session.owned:
            try:
                write_json(run / 'closing-request.json', session.close())
            except Exception as error:
                result.update(status='aborted', neutral_request_error=str(error))
        if reader is not None:
            reader.close()
        if status is not None:
            status.close()
        if args.telemetry_transport == 'tcp' and client is not None:
            try:
                client.close()
                result['tcp_receiver_terminated'] = not client.thread.is_alive()
            except RuntimeError as error:
                result.update(status='aborted', reason=str(error), tcp_receiver_terminated=False)
        write_json(run / 'samples.json', events)
        result.update(actuator_messages_sent=session.submit_attempts if session else 0,
            model_action_requests=0, actor_input=False, foreground_samples=sum(
                e['foreground'] for e in events), samples=len(events))
        write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
