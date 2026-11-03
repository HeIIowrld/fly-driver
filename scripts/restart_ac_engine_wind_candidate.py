"""Publish at most one pinned wind-diagnostic engine restart and observe its outcome.

This does not install Lua, restore race.ini, drive, or declare wind source qualification.
An observed new engine is not proof that the requested wind reached the physics engine.
"""

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.win_process_receipt import ProcessReceipt
from flybrain_drive.wind_locked_restore import inspect_wind_transaction
from flybrain_drive.wind_stimulus_config import isolated_wind_copy
from scripts.control_ac_session import read_session_heartbeat, terminal_receipt, validate_heartbeat


def validate_transaction(manifest, transaction):
    direction, speed = manifest.get('direction_deg'), manifest.get('speed_kmh')
    if (type(direction) is not int or direction not in (0, 90, 180, 270)
            or type(speed) is not int or speed not in (0, 36)
            or speed == 0 and direction != 0):
        raise ValueError('Exact candidate wind stimulus contract required')
    original = (transaction/'original.bin').read_bytes()
    stimulus = (transaction/'stimulus.bin').read_bytes()
    if stimulus != isolated_wind_copy(original, direction_deg=direction, speed_kmh=speed):
        raise ValueError('Saved wind stimulus differs from its declared values')


def validate_restart_intent(intent, request):
    if (intent.get('status') != 'engine_restart_intent_committed'
            or any(intent.get(k) != request[k] for k in
                   ('id', 'command', 'old_engine_pid', 'wind_direction_deg',
                    'wind_speed_kmh', 'original_race_sha256'))
            or intent.get('ui_actions_sent') != 0 or intent.get('actuator_messages_sent') != 0):
        raise ValueError('Original wind restart intent receipt differs')


def validate_new_engine(old, before, old_producer, new_window, state, producer):
    if (old.get('termination_signaled') is not True or old.get('active') is not False
            or new_window.pid == old['pid'] or state.get('generation') == before['generation']
            or int(state['window_handle'].removesuffix('ULL')) != new_window.hwnd
            or state.get('car_id') != 'bmw_z4_gt3' or state.get('track_id') != 'monza'
            or state.get('is_online') is not False or state.get('camera_active') is not False
            or state.get('ai_controlled') is not False or producer == old_producer
            or not producer.startswith('ac-file-'+str(new_window.pid)+'-')):
        raise ValueError('Original process exit and distinct matching engine/producer required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--transaction', required=True)
    parser.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    parser.add_argument('--allow-foreground-engine', action='store_true',
                        help='Allow an engine already foreground; never change window focus')
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    transaction = Path(args.transaction)
    manifest = json.loads((transaction/'manifest.json').read_text(encoding='utf8'))
    validate_transaction(manifest, transaction)
    race = Path(manifest['target'])
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
                                  'arguments': vars(args), 'scope': __doc__})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'request_publications': 0, 'ui_actions_sent': 0,
              'actuator_messages_sent': 0, 'race_file_written': False,
              'wind_source_verified': False, 'restart_outcome_verified': False}
    handle = None
    root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    try:
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original engine required')
        window = windows[0]
        installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        if sha256(installed) != args.candidate_sha256:
            raise ValueError('Exact wind diagnostic candidate required')
        axis = json.loads(Path(args.axis_config).read_text(encoding='utf8'))
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])

        def preflight():
            refuse_leased_transport('wind engine restart', status_path=root/'status.json')
            state, modified, _ = read_session_heartbeat(root/'status.json')
            validate_heartbeat(state, time.time()-modified, mutating=True)
            if (state.get('paused') is not True or state.get('in_menu') is not False
                    or state.get('is_online') is not False
                    or state.get('camera_active') is not False
                    or state.get('ai_controlled') is not False
                    or state.get('car_id') != 'bmw_z4_gt3' or state.get('track_id') != 'monza'
                    or not state.get('capabilities', {}).get('restart_engine')
                    or int(state['window_handle'].removesuffix('ULL')) != window.hwnd
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or api.user.GetForegroundWindow() == window.hwnd
                    and not args.allow_foreground_engine
                    or sdk.GetVJDStatus(axis['device_id']) != 1
                    or sha256(installed) != args.candidate_sha256):
                raise ValueError('Paused background engine and released controller required')
            if sha256(axis['profile_path']) != axis['profile_sha256']:
                raise ValueError('Explicit axis configuration profile hash differs')
            with NativeTeacherPhysicsReader() as reader:
                native = reader.receive()
            fields = native.physics.fields
            if (native.builtin_ai_controlled or abs(fields['speedKmh']) >= .1
                    or fields['gas'] >= .02 or fields['brake'] <= .98):
                raise ValueError('Native same-copy AI off and stationary full brake required')
            inspection = inspect_wind_transaction(race, journal_dir=transaction)
            if (inspection['wind_state'] != 'original_wind'
                    or not inspection['entire_file_matches_original']):
                raise ValueError('Exact original race file required before publication')
            return state, native, inspection

        before, native, inspection = preflight()
        handle = ProcessReceipt(window.pid)
        if not handle.snapshot()['active']:
            raise ValueError('Original process handle must be active')
        producer = FileTelemetryClient(Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry',
                                       timeout=.15, clock=time.perf_counter)
        if not producer.session_id.startswith('ac-file-'+str(window.pid)+'-'):
            raise ValueError('Original producer must match engine PID')
        write_json(run/'before.json', {'session': before, 'window': asdict(window),
            'native': native.authority_record(), 'race_inspection': inspection,
            'producer_session': producer.session_id})
        if not args.execute:
            result['status'] = 'read_only_preflight'
            return
        request = {'schema': 'flybrain.csp.session.request.v1', 'id': uuid.uuid4().hex,
                   'command': 'restart-engine-wind-diagnostic', 'generation': before['generation'],
                   'epoch': before['epoch'], 'expires_clock': before['script_clock']+3,
                   'old_engine_pid': window.pid, 'released_controller_verified': True,
                   'native_ai_off_verified': True,
                   'original_race_sha256': manifest['original_sha256'],
                   'wind_direction_deg': manifest['direction_deg'],
                   'wind_speed_kmh': manifest['speed_kmh']}
        fresh, native, inspection = preflight()
        request = fresh_unpublished_request(before, fresh, request, publication_started=False)
        journal = root/'intent-journal'
        journal.mkdir(exist_ok=True)
        write_json(journal/(request['id']+'.json'), {'request': request, 'before': fresh,
            'native': native.authority_record(), 'race_inspection': inspection,
            'publication_attempts_before_journal': 0})
        write_json(run/'request.json', request)
        result.update(request_publications=1, request_id=request['id'],
                      status='restart_outcome_unknown')
        write_json(run/'result.json', result)
        write_json(root/'request.json', request)
        deadline = time.monotonic()+90
        intent = None
        while time.monotonic() < deadline:
            if intent is None:
                intent = terminal_receipt(root, request)
                if intent is not None:
                    write_json(run/'restart-intent.json', intent)
                    validate_restart_intent(intent, request)
            old = handle.snapshot()
            candidates = [w for w in api.discover() if w.pid != window.pid
                          and Path(w.executable) == Path(window.executable)]
            if intent is not None and not old['active'] and len(candidates) == 1:
                new = candidates[0]
                try:
                    state, modified, _ = read_session_heartbeat(root/'status.json')
                    validate_heartbeat(state, time.time()-modified, mutating=False)
                    new_producer = FileTelemetryClient(
                        Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry', timeout=.15,
                        clock=time.perf_counter)
                    new_producer.snapshot()
                    validate_new_engine(old, before, producer.session_id, new, state,
                                        new_producer.session_id)
                except (OSError, ValueError, TimeoutError, KeyError):
                    time.sleep(.1)
                    continue
                write_json(run/'after.json', {'old_process': old, 'window': asdict(new),
                    'session': state, 'producer_session': new_producer.session_id})
                result.update(status='new_engine_confirmed', restart_outcome_verified=True,
                              new_engine_pid=new.pid, new_generation=state['generation'])
                break
            time.sleep(.1)
        else:
            raise TimeoutError('Original request outcome unknown; inspect, never replay')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        if handle is not None:
            result['original_process'] = handle.snapshot()
            handle.close()
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
