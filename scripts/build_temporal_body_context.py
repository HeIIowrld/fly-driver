"""Build separately timed camera-context and decision body observations offline."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.temporal_body_context import temporal_body_context
from flybrain_drive.virtual_navigation_body import (
    NAVIGATION_FEATURE_NAMES,
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def decode(row):
    raw = base64.b64decode(row['prefix_base64'], validate=True)
    if hashlib.sha256(raw).hexdigest() != row['prefix_sha256']:
        raise ValueError('Original body bytes differ')
    sample = decode_navigation_body_prefix(raw, row['received_monotonic_seconds'])
    if sample.legacy.packet_id != row['packet_id']:
        raise ValueError('Original body packet differs')
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'receipt-audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, audit = Path(args.recording), Path(args.receipt_audit)
    report, manifest = read(audit/'result.json'), read(audit/'manifest.json')['config']
    events_path, samples_path = root/'events.jsonl', root/'control/native-motion-samples.json'
    pairs_path = audit/'receipt-pairs.json'
    if (report['status'] != 'completed_causal_receipt_bound_audit'
            or report['training_admissible'] is not False
            or sha256(events_path) != manifest['events_sha256']
            or sha256(samples_path) != manifest['native_samples_sha256']
            or sha256(pairs_path) != report['pair_sha256']):
        raise ValueError('Unchanged diagnostic receipt audit required')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    pairs, source_rows = read(pairs_path), read(samples_path)
    samples = {}
    for row in source_rows:
        sample = decode(row)
        samples.setdefault(sample.legacy.packet_id, sample)
    if len(events) != len(pairs) or len(events) != report['observations']:
        raise ValueError('Complete event/pair stream required')
    identity = read('configs/ac-virtual-body-v1.json')['native_identity']
    contract = 'configs/ac-virtual-navigation-body-v2.json'
    encoders = [NavigationBodyEncoder(contract, native_identity=identity) for _ in range(2)]
    values, masks, clocks, packets, timing = [], [], [], [], []
    for i, (event, pair) in enumerate(zip(events, pairs, strict=True)):
        if (event['index'] != i or pair['index'] != i
                or pair['camera_capture_id'] != event['camera']['capture_id']):
            raise ValueError('Original event and camera identity required')
        original = decode(event['native'])
        if original.legacy.packet_id != pair['actor_packet']:
            raise ValueError('Recorded decision body differs')
        now = event['decision_monotonic_seconds']
        camera_time = event['camera']['received']
        decision_body = encoders[0].encode(original, episode_id='original-recorded-episode',
                                           decision_monotonic_seconds=now)
        if pair['reason'] == 'prior_receipt_within_host_age':
            prior = samples[pair['prior_packet']]
            if prior.legacy.received_monotonic_seconds != pair['prior_receipt']:
                raise ValueError('Recorded prior receipt differs')
            context_body = encoders[1].encode(prior, episode_id='original-recorded-episode',
                                              decision_monotonic_seconds=camera_time)
            context = temporal_body_context(context_body, decision_body,
                camera_receipt_time=camera_time, render_start_lower=pair['render_start_lower_min'],
                decision_time=now)
            context_values = context.visual_context.values
            context_valid = context.visual_context.valid
            prior_time = prior.legacy.received_monotonic_seconds
            prior_packet = prior.legacy.packet_id
        elif pair['reason'] in ('no_proven_prior_receipt', 'receipt_too_old', 'no_receipts'):
            context_values, context_valid = np.zeros(21), np.zeros(21, dtype=bool)
            prior_time, prior_packet = -1., -1
        else:
            raise ValueError('Unrecognized missing historical observation')
        timing.append([prior_time, pair['render_start_lower_min'], camera_time,
                       original.legacy.received_monotonic_seconds, now])
        values.append([decision_body.values, context_values])
        masks.append([decision_body.valid, context_valid])
        clocks.append(now)
        packets.append([original.legacy.packet_id, prior_packet])
    values, masks, clocks, packets, timing = map(
        np.asarray, (values, masks, clocks, packets, timing))
    if np.any(np.diff(clocks) <= 0):
        raise ValueError('Original decision clock must advance')
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'recording': str(root),
        'events_sha256': sha256(events_path), 'pairs_sha256': sha256(pairs_path),
        'receipt_audit_sha256': sha256(audit/'result.json'), 'contract_sha256': sha256(contract),
        'role': 'separate camera-receipt context and decision-time body; no fit or deployment'})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'body-pairs.npz', values=values, valid=masks,
                        decision_monotonic_seconds=clocks, packets=packets,
                        original_host_timing=timing)
    feature_metrics = {}
    for i, name in enumerate(NAVIGATION_FEATURE_NAMES):
        valid = masks[:, 0, i] & masks[:, 1, i]
        difference = np.abs(values[valid, 0, i]-values[valid, 1, i])
        feature_metrics[name] = {'both_valid': int(valid.sum()),
            'validity_disagrees': int((masks[:, 0, i] != masks[:, 1, i]).sum()),
            'absolute_p50_p95_max': np.percentile(difference, [50, 95, 100]).tolist()
            if len(difference) else None}
    result = {'status': 'completed_temporal_body_context', 'observations': len(events),
              'feature_order': list(NAVIGATION_FEATURE_NAMES), 'features': feature_metrics,
              'body_pairs_sha256': sha256(run/'body-pairs.npz'), 'original_actor_modified': False,
              'camera_time_is_host_receipt_not_exposure': True,
              'context_not_decision_time_body': True,
              'camera_exposure_time_verified': False, 'physics_source_time_verified': False,
              'training_admissible': False, 'optimizer_updates': 0, 'actor_deployed': False}
    write_json(run/'result.json', result)
    print(json.dumps({'status': result['status'], 'observations': len(events),
                      'body_pairs_sha256': result['body_pairs_sha256']}), flush=True)


if __name__ == '__main__':
    main()
