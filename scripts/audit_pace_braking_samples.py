"""Audit moving-brake samples of a declared diagnostic pace task, without fit rights."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'raw-audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source, audit = Path(args.recording), Path(args.raw_audit)
    recording, proof = read(source / 'result.json'), read(audit / 'result.json')
    audit_config = read(audit / 'manifest.json')['config']
    control = read(source / 'control/result.json')
    if (recording['status'] != 'completed' or recording['training_admissible'] is not False
            or recording['teacher_kind'] != 'engine_ai_explicit_pace_task'
            or control['status'] != 'completed'
            or control['pace_task_admission']['natural_ai_labels'] is not False
            or proof['status'] != 'completed' or not proof['camera_aid_state_verified']
            or any(proof['actor_replay_max_errors'].values())
            or audit_config['events_sha256'] != sha256(source / 'events.jsonl')):
        raise ValueError('Completed diagnostic pace task and exact raw audit required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'events_sha256': sha256(source / 'events.jsonl'),
        'raw_audit_sha256': sha256(audit / 'result.json'),
        'control_result_sha256': sha256(source / 'control/result.json'),
        'scope': 'diagnostic native receipt only; no labels or fit', 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for line in (source / 'events.jsonl').read_text(encoding='utf8').splitlines():
        event = json.loads(line)
        raw = base64.b64decode(event['native']['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != event['native']['prefix_sha256']:
            raise ValueError('Original native prefix hash differs')
        sample = decode_physics_prefix(raw, event['native']['received_monotonic_seconds'])
        if sample.packet_id != event['native']['packet_id']:
            raise ValueError('Original packet identity differs')
        authority = event['native_teacher_authority']
        if (authority['packet_id'] != sample.packet_id or not authority['same_native_copy']
                or authority['body_prefix_sha256'] != event['native']['prefix_sha256']):
            raise ValueError('Authority must belong to the same recorded native prefix')
        if sample.fields['brake'] > .05 and sample.fields['speedKmh'] > 1:
            rows.append({'index': event['index'], 'native_packet': sample.packet_id,
                'native_receipt': sample.received_monotonic_seconds,
                'speed_kmh': sample.fields['speedKmh'], 'brake': sample.fields['brake'],
                'same_copy_native_ai': authority['builtin_ai_controlled'],
                'camera_teacher_authority_synchronized': False})
    if not rows:
        raise ValueError('No moving brake observed; retain failure')
    write_json(run / 'moving-brake-receipts.json', rows)
    result = {'status': 'completed_pace_brake_diagnostic_audit', 'moving_brake_count': len(rows),
        'speed_min_kmh': min(row['speed_kmh'] for row in rows),
        'speed_max_kmh': max(row['speed_kmh'] for row in rows),
        'same_copy_ai_true_count': sum(row['same_copy_native_ai'] is True for row in rows),
        'receipt_rows_sha256': sha256(run / 'moving-brake-receipts.json'),
        'external_speed_task': True, 'natural_ai_labels': False,
        'camera_teacher_authority_synchronized': False, 'training_admissible': False,
        'actor_deployed': False}
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
