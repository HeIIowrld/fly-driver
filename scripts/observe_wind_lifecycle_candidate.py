"""One paused diagnostic wind observation from an installed pinned candidate, no control."""

import json
import math
import os
import time
import uuid
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import read_session_heartbeat, terminal_receipt, validate_heartbeat


def validate_observation(receipt, request):
    observation = receipt.get('observation') or {}
    if (receipt.get('status') != 'wind_observed_diagnostic' or receipt.get('id') != request['id']
            or receipt.get('command') != request['command']
            or receipt.get('actor_admissible') is not False
            or receipt.get('ui_actions_sent') != 0 or receipt.get('actuator_messages_sent') != 0
            or observation.get('generation') != request['generation']
            or observation.get('actor_admissible') is not False
            or observation.get('source_verified') is not False):
        raise ValueError('Exact diagnostic-only original wind observation required')
    for name, count in [('wind_velocity_candidate_m_s', 3),
                        ('weather_wind_velocity_candidate_kmh', 2),
                        ('vehicle_velocity_candidate_m_s', 3), ('body_forward_candidate', 3),
                        ('body_left_candidate', 3), ('body_up_candidate', 3)]:
        value = observation.get(name)
        if (not isinstance(value, list) or len(value) != count
                or not all(type(x) in (int, float) and math.isfinite(x) for x in value)):
            raise ValueError('Finite diagnostic vector required: '+name)
    for name in ('weather_wind_speed_candidate_kmh', 'car_physics_timestamp_ms',
                 'started_source_clock', 'finished_source_clock'):
        if (type(observation.get(name)) not in (int, float)
                or not math.isfinite(observation[name])):
            raise ValueError('Finite diagnostic scalar required: '+name)
    if observation['finished_source_clock'] < observation['started_source_clock']:
        raise ValueError('Ordered source clocks required')
    return observation


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--allow-foreground-observation', action='store_true',
                        help='Allow read-only observation if the engine opened in foreground')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    candidate = Path(args.candidate)
    if sha256(candidate) != args.expected_sha256:
        raise ValueError('Exact pinned candidate required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'candidate_sha256': args.expected_sha256, 'scope': __doc__})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'request_publications': 0, 'ui_actions_sent': 0,
              'actuator_messages_sent': 0, 'actor_deployed': False}
    try:
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('One original engine required')
        window = windows[0]
        installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        foreground = api.user.GetForegroundWindow() == window.hwnd
        if (sha256(installed) != args.expected_sha256
                or foreground and not args.allow_foreground_observation):
            raise ValueError('Exact candidate and background engine required')
        result['foreground_at_observation'] = bool(foreground)
        root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
        status_path = root/'status.json'
        refuse_leased_transport('wind diagnostic observation', status_path=status_path)
        before, modified, _ = read_session_heartbeat(status_path)
        validate_heartbeat(before, time.time()-modified, mutating=True)
        if (before.get('paused') is not True or before.get('in_menu') is not False
                or before.get('camera_active') is not False
                or before.get('car_id') != 'bmw_z4_gt3' or before.get('track_id') != 'monza'
                or int(before['window_handle'].removesuffix('ULL')) != window.hwnd):
            raise ValueError('Paused same-engine camera-off state required')
        axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
        if load_vjoy_sdk(axis['dll_path'], axis['dll_sha256']).GetVJDStatus(axis['device_id']) != 1:
            raise ValueError('Released vJoy required')
        with NativeTeacherPhysicsReader() as reader:
            native = reader.receive()
        fields = native.physics.fields
        if (native.builtin_ai_controlled or abs(fields['speedKmh']) >= .1
                or fields['gas'] >= .02 or fields['brake'] <= .98):
            raise ValueError('Native AI off and stationary full brake required')
        write_json(run/'native-before.json', native.authority_record())
        request = {'schema': 'flybrain.csp.session.request.v1', 'id': uuid.uuid4().hex,
                   'command': 'wind-observe-diagnostic', 'generation': before['generation'],
                   'epoch': before['epoch'], 'expires_clock': before['script_clock']+3}
        fresh, modified, _ = read_session_heartbeat(status_path)
        validate_heartbeat(fresh, time.time()-modified, mutating=True)
        request = fresh_unpublished_request(before, fresh, request, publication_started=False)
        journal = root/'intent-journal'
        journal.mkdir(exist_ok=True)
        write_json(journal/(request['id']+'.json'), {'request': request, 'before': fresh,
            'host_clock': time.perf_counter(), 'publication_attempts_before_journal': 0})
        result['request_publications'] = 1  # A publication failure is an unknown outcome.
        write_json(root/'request.json', request)
        write_json(run/'request.json', request)
        deadline = time.monotonic()+5
        receipt = None
        while time.monotonic() < deadline:
            receipt = terminal_receipt(root, request)
            if receipt is not None:
                break
            time.sleep(.02)
        if receipt is None:
            raise TimeoutError('Original request outcome unknown; never automatically retry')
        write_json(run/'api-receipt.json', receipt)
        observation = validate_observation(receipt, request)
        after, modified, _ = read_session_heartbeat(status_path)
        validate_heartbeat(after, time.time()-modified, mutating=True)
        if (after['generation'] != before['generation'] or after['epoch'] != before['epoch']
                or after['paused'] is not True or after['camera_active'] is not False):
            raise ValueError('Same paused camera-off state required after observation')
        write_json(run/'after.json', after)
        result.update(status='completed_wind_diagnostic_observation',
            receipt_sha256=sha256(run/'api-receipt.json'),
            observation=observation, wind_source_verified=False,
            physical_sync_verified=False, exact_physical_time_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
