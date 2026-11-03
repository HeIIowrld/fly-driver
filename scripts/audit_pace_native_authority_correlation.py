"""Correlate earlier raw native AI flag copies with diagnostic event body bytes."""
import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_causal_copy_body_context import first_observed_packets


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'raw-audit', 'native-audit', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root, raw_audit, native_audit = map(Path, (a.recording, a.raw_audit, a.native_audit))
    result, audit, native_proof = (read(root/'result.json'), read(raw_audit/'result.json'),
                                   read(native_audit/'result.json'))
    ac = read(raw_audit/'manifest.json')['config']
    nc = read(native_audit/'manifest.json')['config']
    events_path, copies_path = root/'events.jsonl', root/'native-copy-history.json'
    if (result['status'] != 'completed' or result['training_admissible'] is not False
            or audit['status'] != 'completed' or native_proof['status'] != 'completed'
            or ac['events_sha256'] != sha256(events_path)
            or nc['native_sha256'] != sha256(copies_path)
            or nc['events_sha256'] != sha256(events_path)):
        raise ValueError('Matching completed diagnostic and independent audits required')
    rows = read(copies_path)
    packets = first_observed_packets(rows)
    first = {}
    for row in rows:
        first.setdefault(row['authority']['packet_id'], row)
    if len(first) != len(packets):
        raise ValueError('Unique native copies differ')
    counts, phases, per_event = Counter(), Counter(), []
    for line in events_path.read_text(encoding='utf8').splitlines():
        event = json.loads(line)
        body = event['native']
        raw = base64.b64decode(body['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != body['prefix_sha256']:
            raise ValueError('Event native body differs')
        sample = decode_physics_prefix(raw, body['received_monotonic_seconds'])
        if sample.packet_id != body['packet_id']:
            raise ValueError('Event native packet differs')
        row = first.get(sample.packet_id)
        state = 'missing_packet'
        if row is not None:
            auth = row['authority']
            if auth['body_prefix_sha256'] != body['prefix_sha256']:
                state = 'different_body_bytes'
            elif auth['received_monotonic_seconds'] > body['received_monotonic_seconds']:
                state = 'first_copy_after_event_receipt'
            else:
                state = 'earlier_matching_body_ai_true' if auth['builtin_ai_controlled'] else 'earlier_matching_body_ai_false'
        counts[state] += 1
        f = sample.fields
        moving_brake = f['speedKmh'] > 1 and f['brake'] > .05
        if moving_brake:
            counts['moving_brake_events'] += 1
            counts['moving_brake_'+state] += 1
        phases[event['external_task_at_receipt']['phase']] += 1
        per_event.append(dict(index=event['index'], packet=sample.packet_id, state=state,
                              native_receipt=sample.received_monotonic_seconds,
                              moving_brake=moving_brake, speed_kmh=f['speedKmh'],
                              brake=f['brake'], gas=f['gas'], steer=f['steerAngle']))
    run = begin_run(a.output, dict(program_sha256=sha256(__file__), events_sha256=sha256(events_path),
        copies_sha256=sha256(copies_path), raw_audit_sha256=sha256(raw_audit/'result.json'),
        native_audit_sha256=sha256(native_audit/'result.json'), scope=__doc__))
    write_json(run/'event-correlations.json', per_event)
    out = dict(status='completed_diagnostic_correlation', events=len(per_event), counts=dict(counts),
        phases=dict(phases), correlations_sha256=sha256(run/'event-correlations.json'),
        separate_copy_is_same_copy_authority=False, camera_authority_synchronized=False,
        exclusive_teacher_control_verified=False, physical_timing_verified=False,
        training_admissible=False, role='diagnostic', optimizer_updates=0)
    write_json(run/'result.json', out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
