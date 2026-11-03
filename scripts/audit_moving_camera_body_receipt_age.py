"""Audit recorded moving four-view render lower bounds and body receipt times.

These are host-time upper bounds on render-call age at receipt, not physical
camera exposure or physics sample alignment; this produces no training labels.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'raw-audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source, audit = Path(args.recording), Path(args.raw_audit)
    result = json.loads((source / 'result.json').read_text(encoding='utf8'))
    proof = json.loads((audit / 'result.json').read_text(encoding='utf8'))
    config = json.loads((audit / 'manifest.json').read_text(encoding='utf8'))['config']
    if (result['status'] != 'completed' or proof['status'] != 'completed'
            or not proof['raw_mmf_slots_and_original_metadata_verified']
            or not proof['deferred_persistence_verified']
            or not proof['camera_aid_state_verified']
            or any(proof['actor_replay_max_errors'].values())
            or config['recording_manifest_sha256'] != sha256(source / 'manifest.json')
            or config['events_sha256'] != sha256(source / 'events.jsonl')):
        raise ValueError('Exact raw camera and actor replay audit required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': sha256(source / 'result.json'),
        'events_sha256': sha256(source / 'events.jsonl'),
        'raw_audit_sha256': sha256(audit / 'result.json'),
        'scope': 'render-call host lower bound to native host receipt; not physical sync',
        'training_admissible': False, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    events = [json.loads(line) for line in (source / 'events.jsonl').read_text(
        encoding='utf8').splitlines()]
    if len(events) != proof['decisions'] or len(events) != result['decisions']:
        raise ValueError('All original decision events required')
    rows = []
    for index, event in enumerate(events):
        if event['index'] != index:
            raise ValueError('Original uninterrupted event sequence required')
        native = event['native']
        body_time = native['received_monotonic_seconds']
        decision = event['decision_monotonic_seconds']
        views = event['camera']['views']
        if ({view['view_id'] for view in views} != {'front', 'left', 'right', 'rear'}
                or len(views) != 4 or not all(math.isfinite(x) for x in (body_time, decision))
                or body_time > decision or native['source_time_seconds'] is not None
                or native['simulation_clock_verified'] is not False):
            raise ValueError('Original causal receipt and four-view contract required')
        row = {'index': index, 'native_packet': native['packet_id'],
               'body_receipt': body_time, 'decision': decision,
               'body_receipt_age_ms': 1000 * (decision - body_time), 'views': {}}
        for view in views:
            lower = view['render_started_lower']
            if not math.isfinite(lower) or lower > decision:
                raise ValueError('Causal per-view render lower bound required')
            row['views'][view['view_id']] = {
                'render_lower_host': lower,
                'render_age_upper_at_body_receipt_ms': 1000 * (body_time - lower),
                'render_age_upper_at_decision_ms': 1000 * (decision - lower)}
        rows.append(row)
    report = {}
    for view in ('front', 'left', 'right', 'rear'):
        values = np.asarray([x['views'][view]['render_age_upper_at_body_receipt_ms']
                             for x in rows])
        report[view] = {'minimum_ms': float(values.min()),
                        'median_ms': float(np.median(values)),
                        'p95_ms': float(np.quantile(values, .95)),
                        'maximum_ms': float(values.max()),
                        'negative_upper_bounds': int((values < 0).sum())}
    write_json(run / 'receipt-bounds.json', rows)
    write_json(run / 'result.json', {'status': 'completed_moving_receipt_age_audit',
        'decisions': len(rows), 'views': report,
        'receipt_bounds_sha256': sha256(run / 'receipt-bounds.json'),
        'physical_exposure_sync_verified': False, 'physics_source_time_verified': False,
        'training_admissible': False, 'actor_deployed': False})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
