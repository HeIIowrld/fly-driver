"""Audit per-view causal native receipts; diagnostic only, no physical-time claim."""

import argparse
import base64
import hashlib
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import (
    NAVIGATION_FEATURE_NAMES,
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)


def first_observed_packets(rows):
    """Retain first receipt per ordered packet; never re-date duplicate bytes."""
    packets = []
    previous_end = -1.
    for row in rows:
        a, t = row['authority'], row['copy']
        raw = base64.b64decode(a['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != a['prefix_sha256']:
            raise ValueError('Native raw bytes differ')
        start, end = t['first_copy_start'], t['second_copy_end']
        if not previous_end <= start <= end or end != a['received_monotonic_seconds']:
            raise ValueError('Ordered original copy interval required')
        previous_end = end
        sample = decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size], end)
        if sample.legacy.packet_id != a['packet_id']:
            raise ValueError('Packet ID differs from raw bytes')
        if packets and sample.legacy.packet_id < packets[-1]['packet']:
            raise ValueError('Packet reset requires a new episode')
        if packets and sample.legacy.packet_id == packets[-1]['packet']:
            if a['prefix_sha256'] != packets[-1]['hash']:
                raise ValueError('Repeated packet changed bytes; cannot refresh receipt')
            continue
        packets.append({'packet': sample.legacy.packet_id, 'hash': a['prefix_sha256'],
                        'sample': sample, 'first_end': end, 'first_start': start})
    return packets


def causal_packet_index(receipts, render_lower):
    """Choose only a fully completed copy strictly before the earliest render bound."""
    return bisect_left(receipts, render_lower)-1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.recording)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    native_path, brackets_path = root/'native-copy-windows.json', root/'render-copy-brackets.json'
    metadata_path = root/'diagnostic-frames.json'
    if (result.get('status') != 'completed' or result.get('training_admissible') is not False
            or sha256(native_path) != result['native_copy_sha256']
            or sha256(brackets_path) != result['render_copy_brackets_sha256']
            or sha256(metadata_path) != result['metadata_sha256']):
        raise ValueError('Exact completed diagnostic raw data required')
    native = json.loads(native_path.read_text(encoding='utf8'))
    brackets = json.loads(brackets_path.read_text(encoding='utf8'))
    metadata = json.loads(metadata_path.read_text(encoding='utf8'))
    for i, row in enumerate(metadata):
        if row['index'] != i or sha256(row['source_file']) != row['sha256']:
            raise ValueError('Exact original camera metadata required')
    packets = first_observed_packets(native)
    if len(packets) < 2:
        raise ValueError('At least two distinct packet IDs required')
    receipt = [x['first_end'] for x in packets]
    if any(y <= x for x, y in zip(receipt, receipt[1:], strict=False)):
        raise ValueError('Distinct first packet receipts must advance')
    contract = 'configs/ac-virtual-navigation-body-v2.json'
    identity = json.loads(Path('configs/ac-virtual-body-v1.json').read_text(
        encoding='utf8'))['native_identity']
    encoder = NavigationBodyEncoder(contract, native_identity=identity)
    output_rows, values, valid = [], [], []
    previous_lower = -1.
    for view in brackets:
        lo, hi = view['render_host_bounds']
        if not previous_lower < lo <= hi or view['index'] >= len(metadata):
            raise ValueError('Ordered same-run render bounds required')
        previous_lower = lo
        index = causal_packet_index(receipt, lo)
        if index <= 0:  # Initial packet age has no observed predecessor.
            values.append(np.zeros(21, np.float32))
            valid.append(np.zeros(21, bool))
            output_rows.append({'index': view['index'], 'view': view['view'],
                                'status': 'no_observed_prior_transition'})
            continue
        chosen = packets[index]
        sample = chosen['sample']
        observation = encoder.encode(sample, episode_id='original-diagnostic',
                                     decision_monotonic_seconds=hi)
        values.append(observation.values)
        valid.append(observation.valid)
        output_rows.append({'index': view['index'], 'view': view['view'],
            'status': 'original_first_packet_receipt', 'packet_id': chosen['packet'],
            'first_receipt': chosen['first_end'], 'render_host_bounds': [lo, hi],
            'host_receipt_age_upper_ms': 1000*(hi-chosen['first_end']),
            'valid_count': int(observation.valid.sum())})
    values, valid = np.asarray(values), np.asarray(valid)
    if values.shape != (len(brackets), 21) or valid.shape != values.shape:
        raise ValueError('One explicit 21-channel value and mask per original view required')
    good = [r for r in output_rows if r['status'] == 'original_first_packet_receipt']
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': sha256(root/'result.json'),
        'native_copy_sha256': sha256(native_path), 'brackets_sha256': sha256(brackets_path),
        'metadata_sha256': sha256(metadata_path), 'body_contract_sha256': sha256(contract),
        'scope': __doc__, 'physical_time_or_exposure_verified': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'body-context.npz', values=values, valid=valid)
    write_json(run/'view-receipts.json', output_rows)
    output = {'status': 'completed_causal_copy_body_diagnostic', 'views': len(output_rows),
        'observed_prior_transition_views': len(good),
        'initial_or_missing_views': len(output_rows)-len(good),
        'yaw_speed_both_valid': int((valid[:, 4] & valid[:, 16]).sum()),
        'host_receipt_age_ms_p50_p95_max': np.percentile(
            [x['host_receipt_age_upper_ms'] for x in good], [50, 95, 100]).tolist(),
        'feature_names': list(NAVIGATION_FEATURE_NAMES),
        'body_context_sha256': sha256(run/'body-context.npz'),
        'view_receipts_sha256': sha256(run/'view-receipts.json'),
        'first_receipt_preserved_on_duplicate': True,
        'physical_sample_time_or_exposure_verified': False, 'training_admissible': False,
        'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
