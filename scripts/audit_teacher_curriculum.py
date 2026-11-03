"""Recompute teacher run facts from hashed raw events and verify current stopped state."""
import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader, PHYSICS_LAYOUT, PHYSICS_PREFIX
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root)
    run = begin_run(a.output, dict(arguments=vars(a), seed=780101, actuator_messages_sent=0))
    summaries = []
    for path in sorted(root.glob('live-*')):
        result = json.loads((path/'result.json').read_text(encoding='utf8'))
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf8'))
        if sha256(path/'events.jsonl') != result['events_sha256']:
            raise ValueError('Events hash differs')
        events = [json.loads(x) for x in (path/'events.jsonl').read_text(encoding='utf8').splitlines()]
        feedback = [e for e in events if e['kind'] == 'feedback']
        requests = [e for e in events if e['kind'] == 'teacher_request']
        intervals, old, raw_gear, all_tyres, all_speeds, first_sdk, first_native = [], None, [], [], [], None, None
        for e in events:
            if e['kind'] == 'teacher_history_reset':
                old = None
            if e['kind'] not in ('raw_pair', 'early_source_pair'):
                continue
            for key, data_key, hash_key in (('sdk', 'wire_base64', 'wire_sha256'),
                                          ('native', 'prefix_base64', 'prefix_sha256')):
                raw = base64.b64decode(e[key][data_key], validate=True)
                if hashlib.sha256(raw).hexdigest() != e[key][hash_key]:
                    raise ValueError('Raw payload hash differs')
            if 'graphics' in e:
                timing = e['graphics']
                if hashlib.sha256(base64.b64decode(timing['prefix_base64'])).hexdigest() != timing['prefix_sha256']:
                    raise ValueError('Graphics timing raw hash differs')
            values = iter(PHYSICS_PREFIX.unpack(base64.b64decode(e['native']['prefix_base64'])))
            raw = {name:tuple(next(values) for _ in range(n)) for name, _, n in PHYSICS_LAYOUT}
            all_tyres.append(raw['numberOfTyresOut'][0])
            all_speeds.append(raw['speedKmh'][0])
            if first_native is None:
                first_native = {k:raw[k] for k in ('fuel', 'wheelsPressure', 'tyreCoreTemperature')}
            if e['kind'] == 'early_source_pair':
                continue
            sdk = json.loads(base64.b64decode(e['sdk']['wire_base64']))
            if first_sdk is None:
                first_sdk = sdk
            now = sdk['timestamp_ac']
            if old is not None:
                intervals.append(now-old)
            old = now
            values = iter(PHYSICS_PREFIX.unpack(base64.b64decode(e['native']['prefix_base64'])))
            raw = {name:tuple(next(values) for _ in range(n)) for name, _, n in PHYSICS_LAYOUT}
            raw_gear.append(dict(gear=raw['gear'][0], auto_shifter=raw['autoShifterOn'][0]))
        close = [e for e in events if e['kind'] == 'close']
        if requests and (not close or close[-1]['request']['requested_control'] != dict(
                steering=0., throttle=0., brake=1.)):
            raise ValueError('Final full-brake close receipt absent')
        summaries.append(dict(run=str(path), status=result['status'], reason=result.get('reason'),
            checkpoint=manifest['config']['teacher_configuration']['checkpoint'],
            checkpoint_sha256=manifest['config']['teacher_configuration']['checkpoint_sha256'],
            max_gap_m=manifest['config']['arguments'].get('max_gap_m', 3),
            initial_privileged_pose={k:first_sdk['privileged_teacher']['raw'][k]
                for k in ('world_position_x', 'world_position_y', 'NormalizedSplinePosition')}
                if first_sdk else None,
            initial_native_setup_observation=first_native,
            events_sha256=result['events_sha256'], requests=len(requests), feedback=len(feedback),
            max_speed_kmh=max((e['speed_kmh'] for e in feedback), default=None),
            max_abs_gap_m=max((abs(e['gap_m']) for e in feedback), default=None),
            first_progress=feedback[0]['progress'] if feedback else None,
            last_progress=feedback[-1]['progress'] if feedback else None,
            maximum_tyres_out=max((e['tyres_out'] for e in feedback), default=None),
            maximum_tyres_out_all_raw=max(all_tyres, default=None),
            maximum_native_speed_kmh_all_raw=max(all_speeds, default=None),
            automatic_shift_coast=sum(e.get('automatic_shift_coast', False) for e in requests),
            early_source_pairs=sum(e['kind']=='early_source_pair' for e in events),
            source_interval_min_ms=min(intervals)*1000 if intervals else None,
            source_interval_max_ms=max(intervals)*1000 if intervals else None,
            source_last_interval_ms=intervals[-1]*1000 if intervals else None,
            final_neutral_close_recorded=bool(close), laps=result['laps'],
            native_automatic_shifter_seen=any(x['auto_shifter']==1 for x in raw_gear)))
    manifest = json.loads((root/'live-01/manifest.json').read_text(encoding='utf8'))
    axis = manifest['config']['axis_configuration']
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    snapshots = []
    with NativePhysicsReader(timeout=.15) as reader:
        for _ in range(3):
            sample = reader.receive()
            snapshots.append(physics_record(sample))
            time.sleep(.05)
    hb, modified = read_published_json(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
    released = sdk.GetVJDStatus(axis['device_id']) == 1
    ai_reader = NativeTeacherStatusReader()
    try:
        native_ai = ai_reader.snapshot()
    finally:
        ai_reader.close()
    stopped = released and all(abs(x['fields']['speedKmh']) < .1 and x['fields']['gas'] < .02
        and x['fields']['brake'] > .98 for x in snapshots)
    stopped &= len({x['packet_id'] for x in snapshots}) == len(snapshots)
    stopped &= not native_ai['builtin_ai_controlled'] and not hb['ai_controlled']
    stopped &= not hb['camera_active'] and not hb['expert_probe_active']
    stopped &= 0 <= time.time()-modified <= 2 and not hb.get('error') and not hb.get('pending_receipt_id')
    write_json(run/'result.json', dict(status='completed' if stopped else 'post_stop_failed',
        summaries=summaries, post_stop_verified=stopped, vjoy_released=released,
        post_stop_native=snapshots, session=hb, session_age_seconds=time.time()-modified,
        post_stop_native_ai=native_ai,
        ui_actions_sent=0, actuator_messages_sent=0, student_optimizer_updates=0))
    print(json.dumps(dict(summaries=summaries, post_stop_verified=stopped)))


if __name__ == '__main__':
    main()
