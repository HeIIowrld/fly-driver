"""Reconstruct original prior body receipt at four-view save availability; diagnostic only."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from scripts.audit_causal_copy_body_context import causal_packet_index, first_observed_packets


def select_prior_packet(receipts, lower):
    """Keep first observed packet missing; later packets must be fully received before lower."""
    index = causal_packet_index(receipts, lower)
    return index if index > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'body_audit', 'replay', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), dest=name, required=True)
    a = parser.parse_args()
    root, body, replay = map(Path, (a.recording, a.body_audit, a.replay))
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    source, br, rr = (read(path/'result.json') for path in (root, body, replay))
    bm = read(body/'manifest.json')['config']
    if (source['status'] != 'completed' or source['training_admissible'] is not False
            or bm['recording_result_sha256'] != sha256(root/'result.json')
            or source['native_copy_sha256'] != sha256(root/'native-copy-windows.json')
            or source['metadata_sha256'] != sha256(root/'diagnostic-frames.json')
            or source['render_copy_brackets_sha256'] != sha256(root/'render-copy-brackets.json')
            or br['body_context_sha256'] != sha256(body/'body-context.npz')
            or br['view_receipts_sha256'] != sha256(body/'view-receipts.json')
            or rr['responses_sha256'] != sha256(replay/'responses.npz')):
        raise ValueError('Unchanged original recording, body audit and replay required')
    clock_record = read(root/'clock-after.json')
    clock = CSPClockBounds(clock_record['generation'], maximum_width=.1)
    for p in clock_record['exchanges']:
        clock.observe(p['sent'], p['received'], p['producer_clock'], clock_record['generation'])
    if (abs(clock.low-clock_record['offset_lower_seconds']) > 1e-9
            or abs(clock.high-clock_record['offset_upper_seconds']) > 1e-9
            or clock.high-clock.low > .1):
        raise ValueError('Original exchange-derived clock intersection required')
    packets = first_observed_packets(read(root/'native-copy-windows.json'))
    receipts = [packet['first_end'] for packet in packets]
    frames, brackets = read(root/'diagnostic-frames.json'), read(root/'render-copy-brackets.json')
    earlier = read(body/'view-receipts.json')
    with np.load(replay/'responses.npz', allow_pickle=False) as saved:
        availability = saved['group_source_available_clock'].copy()
    if len(brackets) != len(frames)*4 or len(earlier) != len(brackets) or len(availability) != len(frames):
        raise ValueError('Exact same four-view group and body history required')
    identity = read(Path('configs/ac-virtual-body-v1.json'))['native_identity']
    encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json', native_identity=identity)
    values, masks, records = [], [], []
    for i, row in enumerate(frames):
        source_file = Path(row['source_file'])
        if row['index'] != i or sha256(source_file) != row['sha256']:
            raise ValueError('Original indexed camera metadata required')
        frame = read(source_file)
        if (frame['index'] != i or frame['request_id'] != source['request_id']
                or row['diagnostic']['generation'] != clock.generation):
            raise ValueError('Exact recorded camera generation required')
        view = {item['view_id']: item for item in frame['views']}
        if set(view) != set(VIEW_ORDER) or len(frame['views']) != 4:
            raise ValueError('Four distinct original views required')
        for j, name in enumerate(VIEW_ORDER):
            b = brackets[i*4+j]
            expected = view[name]['render_started_clock'] + clock.low
            if b['index'] != i or b['view'] != name or abs(b['render_host_bounds'][0]-expected) > 1e-8:
                raise ValueError('Clock bounds must match hashed original render bounds')
        available = max(view[name]['save_finished_clock'] for name in VIEW_ORDER)
        if available != availability[i]:
            raise ValueError('Exact FlyVis original group availability required')
        low, high = available+clock.low, available+clock.high
        index = select_prior_packet(receipts, low)
        if index is None:
            values.append(np.zeros(21, np.float32))
            masks.append(np.zeros(21, bool))
            records.append({'index': i, 'status': 'no_observed_prior_transition',
                            'availability_host_bounds': [low, high]})
            continue
        packet = packets[index]
        observed = encoder.encode(packet['sample'], episode_id='original-diagnostic',
                                  decision_monotonic_seconds=high)
        values.append(observed.values)
        masks.append(observed.valid)
        old = earlier[i*4]
        records.append({'index': i, 'status': 'first_receipt_before_availability_bound',
            'availability_host_bounds': [low, high], 'first_receipt': packet['first_end'],
            'packet_id': packet['packet'], 'front_render_packet_id': old.get('packet_id'),
            'host_receipt_age_upper_ms': 1000*(high-packet['first_end'])})
    values, masks = np.asarray(values), np.asarray(masks)
    with np.load(body/'body-context.npz', allow_pickle=False) as data:
        front, front_valid = data['values'].reshape(-1, 4, 21)[:, 0], data['valid'].reshape(-1, 4, 21)[:, 0]
    common = masks[:, 4] & front_valid[:, 4]
    good = [row for row in records if row['status'] != 'no_observed_prior_transition']
    different = [row for row in good if row['front_render_packet_id'] is not None
                 and row['packet_id'] != row['front_render_packet_id']]
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': sha256(root/'result.json'),
        'body_audit_sha256': sha256(body/'result.json'), 'flyvis_sha256': rr['responses_sha256'],
        'clock_after_sha256_at_audit': sha256(root/'clock-after.json'),
        'clock_contract': 'unit-rate assumption; checked against original pings and render bounds',
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'availability-body.npz', values=values, valid=masks,
                        group_source_available_clock=availability)
    write_json(run/'receipts.json', records)
    result = {'status': 'completed_availability_body_diagnostic', 'groups': len(frames),
        'selected_groups': len(good), 'missing_groups': len(frames)-len(good),
        'different_from_front_render_packet': len(different),
        'host_receipt_age_ms_p50_p95_max': np.percentile(
            [x['host_receipt_age_upper_ms'] for x in good], [50, 95, 100]).tolist(),
        'common_yaw_groups': int(common.sum()), 'yaw_abs_difference_max_rad_s': float(
            np.max(np.abs(values[common, 4]-front[common, 4]))) if common.any() else None,
        'arrays_sha256': sha256(run/'availability-body.npz'),
        'receipts_sha256': sha256(run/'receipts.json'),
        'clock_after_hash_originally_sealed': False, 'physical_sample_or_exposure_time_verified': False,
        'group_availability_is_actual_actor_decision': False, 'training_admissible': False,
        'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
