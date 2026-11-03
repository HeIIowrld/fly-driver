"""Bounded stationary native/SDK steering calibration through vJoy; never changes focus."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def validate_stationary_calibration_state(status, age, flag, fields, generation):
    if (not np.isfinite(age) or not 0 <= age <= 2 or status.get('generation') != generation
            or status.get('paused') is not False or status.get('in_menu') is not False
            or status.get('is_online') is not False or status.get('ai_controlled') is not False
            or status.get('camera_active') is not False
            or status.get('expert_probe_active') is not False
            or status.get('pending_receipt_id') or status.get('error')
            or flag.get('builtin_ai_controlled') is not False
            or not np.isfinite([fields['speedKmh'], fields['gas'], fields['brake']]).all()
            or abs(fields['speedKmh']) >= .1 or fields['gas'] >= .02 or fields['brake'] <= .98):
        detail = dict(age_seconds=age, generation=status.get('generation'), expected=generation,
            flags={k: status.get(k) for k in ('paused', 'in_menu', 'is_online', 'ai_controlled',
                                            'camera_active', 'expert_probe_active',
                                            'pending_receipt_id', 'error')},
            native_ai=flag.get('builtin_ai_controlled'),
            body={k: fields[k] for k in ('speedKmh', 'gas', 'brake')})
        raise ValueError('Stationary calibration state rejected: '+json.dumps(detail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ac-axis-probe.json')
    parser.add_argument('--amplitude', type=float, default=.1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not np.isfinite(args.amplitude) or not 0 < args.amplitude <= 1:
        raise ValueError('Finite normalized stationary steering amplitude in (0,1] required')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    run = begin_run(args.output, dict(scope=__doc__, configuration=config,
        steering_requests=[0, args.amplitude, -args.amplitude, 0], samples_per_stage=8))
    session, records, sessions, ai = None, [], set(), None
    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1 or sha256(config['profile_path']) != config['profile_sha256']:
            raise ValueError('Original engine and exact reviewed profile required')
        identity = windows[0]
        status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        first_status, _ = read_published_json(status_path)
        generation = first_status['generation']
        if (first_status.get('car_id') != config['car_model']
                or first_status.get('track_id') != config['track']):
            raise ValueError('Reviewed offline car and track required')
        ai = NativeTeacherStatusReader()
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                      timeout=.5)
        with NativePhysicsReader(timeout=.5) as physics:
            def guard():
                status, modified = read_published_json(status_path)
                flag, sample = ai.snapshot(), physics.receive()
                validate_stationary_calibration_state(status, time.time()-modified, flag,
                                                       sample.fields, generation)
                if (native.probe(identity.hwnd, require_foreground=False) != identity
                        or int(status['window_handle'].removesuffix('ULL')) != identity.hwnd
                        or sha256(config['profile_path']) != config['profile_sha256']):
                    raise ValueError('Engine/profile changed')
                return flag

            guard()
            sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
            session = VJoySession(sdk, config['device_id'],
                                  [AxisBinding(**b) for b in config['bindings']])
            opening = session.open()
            with (run / 'samples.jsonl').open('x', encoding='utf8') as stream:
                stream.write(json.dumps(dict(kind='open', request=opening)) + '\n')
                for stage, steering in enumerate([0., args.amplitude, -args.amplitude, 0.]):
                    for index in range(8):
                        before_ai = guard()
                        request = session.submit(steering, 0., 1.)
                        packet, sample = client.receive(), physics.receive()
                        if not packet.raw['producer_session_id'].startswith(
                                'ac-file-'+str(identity.pid)+'-'):
                            raise ValueError('SDK packet belongs to a different engine')
                        after_ai = ai.snapshot()
                        fresh, modified = read_published_json(status_path)
                        try:
                            validate_stationary_calibration_state(fresh, time.time()-modified,
                                after_ai, sample.fields, generation)
                        except ValueError:
                            write_json(run/'rejected-after-request.json', dict(stage=stage,
                                index=index, request=request, status=fresh, ai=after_ai,
                                native=physics_record(sample)))
                            raise
                        sessions.add(packet.raw['producer_session_id'])
                        row = dict(stage=stage, index=index, request=request,
                            native_ai_before=before_ai, native_ai_after=after_ai,
                            sdk=record_packet(packet), native=physics_record(sample),
                            sdk_steer_raw=packet.raw['steerAngle'],
                            was_foreground_at_record=bool(
                                native.user.GetForegroundWindow() == identity.hwnd))
                        records.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                        if abs(sample.fields['speedKmh']) > 2:
                            raise RuntimeError('Stationary speed budget exceeded')
                        time.sleep(.05)
            closing = session.close()
            write_json(run / 'closing-request.json', closing)
        medians = [float(np.median([r['sdk_steer_raw'] for r in records
                                   if r['stage'] == stage and r['index'] >= 2]))
                   for stage in range(4)]
        gain = (medians[1] - medians[2]) / (2 * args.amplitude)
        if len(sessions) != 1:
            raise ValueError('SDK engine session changed during calibration')
        write_json(run / 'result.json', dict(status='completed', samples=len(records),
            profile_sha256=config['profile_sha256'], dll_sha256=config['dll_sha256'],
            device_id=config['device_id'], bindings=config['bindings'],
            producer_session_id=sessions.pop(),
            request_amplitude=args.amplitude,
            sdk_steering_stage_medians=medians, sdk_raw_per_normalized_request=gain,
            foreground_records=sum(r['was_foreground_at_record'] for r in records),
            actuator_messages_sent=session.submit_attempts,
            source_units_independently_verified=False,
            moving_driving_verified=False, goal_achieved=False))
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', samples=len(records),
            error=type(error).__name__, message=str(error),
            actuator_messages_sent=session.submit_attempts if session else 0, goal_achieved=False))
        raise
    finally:
        if session is not None and session.owned:
            session.close()
        if ai is not None:
            ai.close()


if __name__ == '__main__':
    main()
