"""Estimate publication windows conditionally; original read-start times are unavailable."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def conditional_window(previous, receipt, render_lower, decision):
    if (not all(np.isfinite(x) for x in (previous, receipt, render_lower, decision))
            or not previous < receipt < render_lower <= decision):
        raise ValueError('Receipt and render ordering differs')
    return {'publication_window_width_ms': 1000*(receipt-previous),
            'decision_minus_receipt_ms': 1000*(decision-receipt),
            'conditional_decision_since_publication_upper_ms': 1000*(decision-previous)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--receipt-audit', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, audit = Path(args.recording), Path(args.receipt_audit)
    report, cfg = read(audit/'result.json'), read(audit/'manifest.json')['config']
    samples_path, events_path = root/'control/native-motion-samples.json', root/'events.jsonl'
    pairs_path = audit/'receipt-pairs.json'
    if (report.get('status') != 'completed_causal_receipt_bound_audit'
            or sha256(samples_path) != cfg['native_samples_sha256']
            or sha256(events_path) != cfg['events_sha256']
            or sha256(pairs_path) != report['pair_sha256']):
        raise ValueError('Unchanged original receipt audit required')
    original = read(samples_path)
    samples = []
    for row in original:
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != row['prefix_sha256']:
            raise ValueError('Original native prefix hash differs')
        sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
        if sample.packet_id != row['packet_id']:
            raise ValueError('Native packet differs from original bytes')
        if samples and sample.received_monotonic_seconds <= samples[-1].received_monotonic_seconds:
            raise ValueError('Original receipt clock must strictly advance')
        if samples and sample.packet_id < samples[-1].packet_id:
            raise ValueError('Packet reset requires explicit new episode')
        samples.append(sample)
    first = {}
    for index, sample in enumerate(samples):
        first.setdefault(sample.packet_id, index)
    pairs = read(pairs_path)
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    if len(events) != len(pairs) or len(events) != report['observations']:
        raise ValueError('Original event count differs')
    rows = []
    for event, pair in zip(events, pairs, strict=True):
        if event['index'] != pair['index']:
            raise ValueError('Original decision identity differs')
        packet = pair['prior_packet']
        if packet is None:
            rows.append({'index': event['index'], 'status': 'no_prior_packet'})
            continue
        index = first[packet]
        receipt = samples[index].received_monotonic_seconds
        if receipt != pair['prior_receipt']:
            raise ValueError('Selected original receipt differs')
        if index == 0 or samples[index-1].packet_id == packet:
            rows.append({'index': event['index'], 'status': 'no_previous_distinct_observation'})
            continue
        previous = samples[index-1].received_monotonic_seconds
        decision = event['decision_monotonic_seconds']
        bounds = conditional_window(previous, receipt, pair['render_start_lower_min'], decision)
        rows.append({'index': event['index'], 'status': 'conditional_window',
            'packet': packet, 'previous_distinct_receipt': previous, 'first_receipt': receipt,
            'decision': decision, **bounds})
    good = [row for row in rows if row['status'] == 'conditional_window']
    if not good:
        raise ValueError('No bounded publication candidates')
    def quantiles(key):
        return np.percentile([row[key] for row in good], [50, 95, 100]).tolist()
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'events_sha256': sha256(events_path), 'native_samples_sha256': sha256(samples_path),
        'receipt_audit_sha256': sha256(audit/'result.json'), 'pairs_sha256': sha256(pairs_path),
        'assumption': ('coherent monotonic publication and previous copied packet remains current '
                       'at its post-copy receipt timestamp; latter is unverified'),
        'not_an_assumption': 'physical sampling happened at publication', 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'publication-windows.json', rows)
    output = {'status': 'completed_conditional_publication_window_audit',
        'decisions': len(rows), 'conditional_windows': len(good),
        'publication_window_ms_p50_p95_max': quantiles('publication_window_width_ms'),
        'decision_minus_receipt_ms_p50_p95_max': quantiles('decision_minus_receipt_ms'),
        'conditional_publication_age_ms_p50_p95_max': quantiles(
            'conditional_decision_since_publication_upper_ms'),
        'conditional_publication_age_over_50ms': sum(
            row['conditional_decision_since_publication_upper_ms'] > 50 for row in good),
        'publication_age_upper_bound_proven': False,
        'physical_sample_age_upper_bound_proven': False, 'camera_exposure_time_verified': False,
        'training_admissible': False, 'actor_deployed': False,
        'rows_sha256': sha256(run/'publication-windows.json')}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
