"""Explicit CSP session lifecycle requests; never activates a window or injects keys."""

import argparse
import math
import os
import time
import uuid
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_native_physics import NativePhysicsReader, repeated_snapshot
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.teacher_pace import PACE_COMMAND, PACE_REQUEST, validate_pace_receipt
from scripts.record_ac_wgc import physics_record


def read(path):
    return read_published_json(path)[0]


def read_session_heartbeat(path, *, maximum_wait=.5):
    """Wait for lifecycle metadata only, never replay an API request or cache old state.

    CSP publishes every250ms. A failed file replacement can leave the pathname
    absent until the next heartbeat; the10ms actor reader budget is unrelated.
    The returned timestamp belongs to the same opened file as the returned bytes.
    """
    if not 0 < maximum_wait <= .5:
        raise ValueError('Lifecycle metadata wait is bounded to500ms')
    deadline = time.monotonic()+maximum_wait
    retries = 0
    while True:
        try:
            value, modified = read_published_json(path)
            return value, modified, retries
        except (FileNotFoundError, PermissionError) as error:
            transient = (isinstance(error, FileNotFoundError)
                         or getattr(error, 'winerror', None) in (32, 33)
                         or os.name == 'nt' and error.errno == 13)
            if not transient or time.monotonic() >= deadline:
                raise
            retries += 1
            time.sleep(.005)


def validate_heartbeat(status, age_seconds, *, mutating):
    if not math.isfinite(age_seconds) or not 0 <= age_seconds <= 2:
        raise ValueError("Fresh session API heartbeat required")
    if mutating and (status.get("error") or status.get("pending_receipt_id")):
        raise ValueError("Error-free session API heartbeat required for mutations")
    if mutating and (status.get('expert_probe_active') or status.get('ai_controlled')):
        raise ValueError('AI/probe must be inactive before a new mutation')


def validate_expert_probe(status, *, motion=False, collection=False, fast=False, recovery=False,
                          lateral=False, warm20=False, brake_pulse=False, speed60=False,
                          pace=False):
    capability = 'expert_collect30' if collection else (
        'expert_motion_probe' if motion else 'expert_probe')
    if fast:
        capability = 'expert_collect30_speed30' if collection else 'expert_motion_speed30'
    if speed60:
        capability = 'expert_collect30_speed60' if collection else 'expert_motion_speed60'
    if recovery:
        capability = 'expert_speed20_recovery15' if warm20 else 'expert_speed_recovery15'
    if lateral:
        capability = 'expert_lateral_recovery15'
    if brake_pulse:
        capability = 'expert_brake_pulse30'
    if pace:
        capability = 'expert_pace60_100_stop30'
    if (status.get('paused') is not True or status.get('ai_controlled') is not False
            or status.get('physics_api_allowed') is not True
            or status.get('capabilities', {}).get(capability) is not True):
        raise ValueError('Paused AI-off session with explicit expert probe capability required')


def validate_probe_receipt(receipt, *, motion=False, collection=False, fast=False, warm20=False,
                           speed60=False, pace=False):
    delta = receipt.get('maximum_position_delta_m')
    pass_key = 'motion_probe_passed' if motion else 'stationary_probe_passed'
    limit = (360 if fast else 180) if collection else (40 if fast else 20)
    if speed60:
        if not motion or not fast or warm20:
            raise ValueError('Separate60kmh profile required')
        limit = 720 if collection else 80
    if pace:
        if not (motion and collection and fast) or speed60 or warm20:
            raise ValueError('Separate explicit pace task envelope required')
        limit = 1100
    if warm20:
        if not (motion and collection and fast):
            raise ValueError('20-to15 recovery requires the explicit30second profile')
        limit = 220
    if (receipt.get(pass_key) is not True
            or receipt.get('probe_seen_enabled') is not True
            or not isinstance(delta, (int, float)) or not math.isfinite(delta)
            or not (0.1 < delta <= limit
                    if motion else 0 <= delta <= .02)):
        raise ValueError('AI enabled/off probe was not verified within its movement contract')
    if motion:
        speed = receipt.get('maximum_speed_kmh')
        samples = receipt.get('telemetry', [])
        if (not isinstance(speed, (int, float)) or not math.isfinite(speed)
                or not 0 <= speed <= (120 if pace else 80 if speed60 else (
                    25 if warm20 else (40 if fast else 20)))
                or len(samples) < 2
                or not any(s.get('ai_controlled') is True and not s.get('paused')
                           and isinstance(s.get('gas'), (int, float)) and s['gas'] > .01
                           for s in samples)):
            raise ValueError('Moving engine-AI effective throttle was not observed')


def terminal_receipt(root, request):
    # Accepted file is immutable in v2; the legacy final file remains readable.
    for suffix in ('.terminal.json', '.json'):
        try:
            receipt = read(root / (request['id'] + suffix))
        except FileNotFoundError:
            continue
        if receipt.get('id') != request['id']:
            raise ValueError('Lifecycle receipt ID differs from the owned request')
        if receipt['status'] == 'accepted':
            continue
        if receipt['status'].startswith(('api_returned_', 'api_outcome_')) and (
                receipt.get('command') != request['command']):
            raise ValueError('Lifecycle receipt command differs from the owned request')
        return receipt
    return None


def validate_lateral_recovery_receipt(receipt):
    if (receipt.get('lateral_recovery15_passed') is not True
            or receipt.get('before', {}).get('wheels_outside') != 1
            or receipt.get('after', {}).get('wheels_outside') != 0
            or receipt.get('last_driving_wheels_outside') != 0
            or receipt.get('maximum_position_delta_m', 0) <= 5
            or not any(s.get('ai_controlled') is True and s.get('paused') is False
                       and s.get('phase') == 'driving' and s.get('wheels_outside') == 0
                       and s.get('speed_kmh', 0) > 1 for s in receipt.get('telemetry', []))):
        raise ValueError('Observed AI recovery from one outside tyre to zero was not verified')


def target_reached(command, before, after):
    if after['generation'] != before['generation']:
        raise ValueError('Lifecycle app generation changed; outcome requires inspection')
    if command == 'restart':
        return after['epoch'] > before['epoch'] and after['in_menu']
    if command == 'start':
        return not after['in_menu'] and after['live']
    if command == 'disable-ideal-line':
        return (after.get('ideal_line_shown') is False
                and after.get('paused') == before.get('paused')
                and after.get('ai_controlled') is False and not after.get('camera_active'))
    if command in ('expert-probe', 'expert-motion-probe', 'expert-collect30',
                   'expert-motion-speed30', 'expert-collect30-speed30', 'expert-speed-recovery15',
                   'expert-lateral-recovery15', 'expert-speed20-recovery15',
                   'expert-brake-pulse30', 'expert-motion-speed60', 'expert-collect30-speed60',
                   PACE_COMMAND):
        return (after.get('paused') is True and after.get('ai_controlled') is False
                and after.get('expert_probe_active') is False)
    return after['paused'] == (command == 'pause')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', choices=['status', 'restart', 'start', 'pause', 'resume',
                                            'disable-ideal-line',
                                            'expert-probe', 'expert-motion-probe',
                                            'expert-collect30', 'expert-motion-speed30',
                                            'expert-collect30-speed30', 'expert-speed-recovery15',
                                            'expert-lateral-recovery15',
                                            'expert-speed20-recovery15', 'expert-brake-pulse30',
                                            'expert-motion-speed60', 'expert-collect30-speed60',
                                            PACE_COMMAND],
                        default='status')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-live-aid-command', action='store_true',
                        help='Ideal-line command only, with verified stationary native brake hold')
    parser.add_argument('--camera-request-id',
                        help='Motion probe only: pin an already owned camera')
    parser.add_argument('--ai-steer-multiplier', type=float, choices=[1., .9], default=1.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.allow_live_aid_command and args.command != 'disable-ideal-line':
        raise ValueError('Live aid option is limited to the ideal-line command')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862830,
                                     ui_actions_sent=0, actuator_messages_sent=0))
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
                  lifecycle_requests_sent=0, goal_achieved=False)
    warm20 = args.command == 'expert-speed20-recovery15'
    brake_pulse = args.command == 'expert-brake-pulse30'
    speed60 = args.command in ('expert-motion-speed60', 'expert-collect30-speed60')
    pace = args.command == PACE_COMMAND
    recovery = args.command == 'expert-speed-recovery15' or warm20
    lateral = args.command == 'expert-lateral-recovery15'
    fast = (args.command in ('expert-motion-speed30', 'expert-collect30-speed30')
            or recovery or brake_pulse or speed60 or pace)
    collection = (args.command in ('expert-collect30', 'expert-collect30-speed30')
                  or recovery or lateral or brake_pulse
                  or args.command == 'expert-collect30-speed60' or pace)
    motion = args.command == 'expert-motion-probe' or collection or fast
    probe = args.command == 'expert-probe' or motion
    motion_seconds = 30 if collection else 3
    native_observer = None
    try:
        root = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-session'
        before, modified, retries = read_session_heartbeat(root / 'status.json')
        result['heartbeat_metadata_retries'] = retries
        validate_heartbeat(before, time.time()-modified,
                           mutating=args.execute and args.command != 'status')
        native = WindowsACWindow()
        windows = native.discover()
        if (len(windows) != 1 or int(before['window_handle'].removesuffix('ULL')) != windows[0].hwnd
                or before['car_id'] != 'bmw_z4_gt3' or before['track_id'] != 'monza'
                or before['is_online']):
            raise ValueError('Reviewed offline original AC engine identity required')
        result['foreground_before'] = bool(native.user.GetForegroundWindow() == windows[0].hwnd)
        deployed = Path(windows[0].executable).parent / 'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        source = Path('integrations/ac_csp/candidates'
                      '/FlyBrainCamera-airflow-motion-diagnostic-v1.lua')
        if sha256(deployed) != sha256(source):
            raise ValueError('Lifecycle app source/deployment differ')
        write_json(run / 'before.json', before)
        if args.command == 'status' or not args.execute:
            result.update(status='read_only', session=before)
        else:
            if args.command == 'disable-ideal-line' and (
                    type(before.get('ideal_line_shown')) is not bool
                    or not (before['paused'] or args.allow_live_aid_command and before['live'])
                    or before.get('capabilities', {}).get('disable_ideal_line') is not True):
                raise ValueError('Paused known ideal-line state and explicit sim API required')
            if probe:
                validate_expert_probe(before, motion=motion, collection=collection,
                                      fast=fast, recovery=recovery, lateral=lateral, warm20=warm20,
                                      brake_pulse=brake_pulse, speed60=speed60, pace=pace)
                if fast and args.ai_steer_multiplier != .9:
                    raise ValueError('30kmh expert profile requires bounded AI steer .9')
                if collection and (args.ai_steer_multiplier != .9 or not args.camera_request_id):
                    raise ValueError('30-second collection requires owned camera and AI steer .9')
            axis = read(Path('configs/ac-axis-probe.json'))
            sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
            owns_camera = (motion and args.camera_request_id is not None
                           and before.get('camera_request_id') == args.camera_request_id)
            if (sdk.GetVJDStatus(axis['device_id']) != 1
                    or before['camera_active'] and not owns_camera
                    or args.camera_request_id is not None and not owns_camera):
                raise ValueError('Controller must be released and camera acquisition terminal')
            with NativePhysicsReader(timeout=.15) as reader:
                sample = reader.receive()
            write_json(run / 'native-before.json', physics_record(sample))
            if (abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
                    or sample.fields['brake'] <= .98):
                raise ValueError('Actual stationary brake hold required before lifecycle command')
            request = dict(schema='flybrain.csp.session.request.v1', id=uuid.uuid4().hex,
                command=args.command, generation=before['generation'], epoch=before['epoch'],
                expires_clock=before['script_clock']+3)
            if args.allow_live_aid_command:
                request['allow_live_stationary'] = True
            if probe:
                request.update(speed_limit_kmh=(100 if pace else 60 if speed60 else
                                               (20 if warm20 else (30 if fast else 15))),
                               hold_seconds=motion_seconds if motion else .5)
            if motion:
                request['ai_steer_multiplier'] = args.ai_steer_multiplier
            if recovery:
                request.update(recovery_target_kmh=15, warmup_seconds=3)
            if brake_pulse:
                request.update(brake_pulse_seconds=.4, brake_pulse_after_seconds=3)
            if pace:
                request.update(PACE_REQUEST)
            if lateral:
                request['lateral_recovery'] = True
            if owns_camera:
                request['camera_request_id'] = args.camera_request_id
            if motion:
                native_observer = NativePhysicsReader(timeout=.02)
                result.update(training_admissible=False,
                              observation_role='raw motion probe, no synchronized actor labels')
            # NAS artifact writes above must not consume this never-published ID's TTL.
            # Journal locally before the one publication; issued IDs are never renewed.
            journal = root/'intent-journal'
            journal.mkdir(exist_ok=True)
            fresh, modified, retries = read_session_heartbeat(root/'status.json')
            result['heartbeat_metadata_retries'] += retries
            validate_heartbeat(fresh, time.time()-modified, mutating=True)
            request = fresh_unpublished_request(before, fresh, request,
                publication_started=bool(result['lifecycle_requests_sent']))
            intent_path = journal/(request['id']+'.json')
            write_json(intent_path, dict(request=request, before=fresh,
                original_before_script_clock=before['script_clock'],
                host_clock=time.perf_counter(), deadline_seconds=3,
                publication_attempts_before_journal=0))
            result['lifecycle_publication_attempts'] = 1
            write_json(root / 'request.json', request)
            result['lifecycle_requests_sent'] = 1
            result['local_intent_journal'] = str(intent_path)
            result['local_intent_journal_sha256'] = sha256(intent_path)
            write_json(run / 'request.json', request)
            write_json(run / 'publication-before.json', fresh)
            deadline = time.monotonic()+(motion_seconds+(
                11 if speed60 or pace else 7) if motion else 5)
            terminal = None
            native_samples = []
            while time.monotonic() < deadline:
                if native_observer:
                    native_samples.append(physics_record(repeated_snapshot(
                        native_observer.mapping, clock=time.perf_counter)))
                try:
                    receipt = terminal_receipt(root, request)
                    if receipt is not None:
                        terminal = receipt
                        break
                except (FileNotFoundError, PermissionError):
                    pass
                time.sleep(.02)
            if motion:
                write_json(run / 'native-motion-samples.json', native_samples)
                result['native_motion_receipt_clock'] = 'host_monotonic_QPC_time.perf_counter'
            if terminal is None:
                raise TimeoutError('Same request outcome unknown; do not automatically retry')
            write_json(run / 'api-receipt.json', terminal)
            if probe:
                result['ai_api_calls'] = terminal.get('api_calls', [])
            if terminal['status'] != 'api_returned_true':
                raise ValueError('Session API refused request: '+str(terminal))
            if probe:
                validate_probe_receipt(terminal, motion=motion, collection=collection,
                                       fast=fast, warm20=warm20, speed60=speed60, pace=pace)
                if pace:
                    result['pace_task_admission'] = validate_pace_receipt(terminal)
                if speed60:
                    result['speed60_target_reached'] = terminal['maximum_speed_kmh'] >= 50
                    if collection and not result['speed60_target_reached']:
                        raise ValueError('60kmh collection did not reach50kmh; retain failure')
                if recovery and terminal.get('recovery15_passed') is not True:
                    raise ValueError('Overspeed recovery under AI was not verified')
                if brake_pulse and terminal.get('brake_pulse_passed') is not True:
                    raise ValueError('Actual finite AI brake pulse response was not verified')
                if lateral:
                    validate_lateral_recovery_receipt(terminal)
            reached = False
            while time.monotonic() < deadline:
                try:
                    after, modified, retries = read_session_heartbeat(root / 'status.json')
                    result['heartbeat_metadata_retries'] += retries
                    validate_heartbeat(after, time.time()-modified, mutating=False)
                    reached = target_reached(args.command, before, after)
                    if reached:
                        write_json(run / 'after.json', after)
                        break
                except (FileNotFoundError, PermissionError):
                    pass
                time.sleep(.02)
            if not reached:
                raise TimeoutError('API returned true but target state was not observed')
            if probe:
                with NativePhysicsReader(timeout=.15) as reader:
                    final_sample = reader.receive()
                write_json(run / 'native-after.json', physics_record(final_sample))
                if (abs(final_sample.fields['speedKmh']) >= .1
                        or final_sample.fields['gas'] >= .02
                        or final_sample.fields['brake'] <= .98):
                    raise ValueError('Independent native stationary brake hold not verified')
            result.update(status='completed', target_state_observed=True,
                foreground_after=bool(native.user.GetForegroundWindow() == windows[0].hwnd))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if native_observer:
            native_observer.close()
    write_json(run / 'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
