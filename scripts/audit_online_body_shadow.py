"""Independently audit online camera/body shadow against retained native copy receipts."""

import argparse
import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from scripts.audit_causal_copy_body_context import first_observed_packets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.recording)
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    result = read(root/'result.json')
    files = ((root/'native-copy-windows.json', result['native_copy_sha256']),
             (root/'online-body-shadow.json', result['shadow_sha256']),
             (root/'diagnostic-frames.json', result['metadata_sha256']))
    if (result['status'] != 'completed' or result['training_admissible'] is not False
            or result['shadow_error'] is not None
            or any(sha256(path) != expected for path, expected in files)):
        raise ValueError('Exact completed original online diagnostic required')
    native, events, frames = (read(path) for path, _ in files)
    if len(events) != len(frames) or len(events) != result['frames']:
        raise ValueError('All original camera frames must have an online event')
    packets = first_observed_packets(native)
    receipts = [packet['first_end'] for packet in packets]
    identity = read(Path('configs/ac-virtual-body-v1.json'))['native_identity']
    encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                       native_identity=identity) for _ in range(5)]
    recorded_clock = read(root/'clock-before.json')
    low = recorded_clock['offset_lower_seconds']
    names = ('front', 'left', 'right', 'rear')
    missing, valid, yaw, build_ms, camera_age_ms, counts = 0, 0, 0, [], [], []
    last_decision, last_count = -1., 0
    for i, (event, frame_row) in enumerate(zip(events, frames, strict=True)):
        path = Path(frame_row['source_file'])
        if (frame_row['index'] != i or event['index'] != i
                or sha256(path) != frame_row['sha256'] or event['metadata_sha256'] != frame_row['sha256']):
            raise ValueError('Exact original frame identity and metadata required')
        frame = read(path)
        views = {v['view_id']: v for v in frame['views']}
        receipt, decision, count = event['camera_receipt'], event['decision'], event['native_rows_consumed']
        if (set(views) != set(names) or not last_decision < receipt <= decision
                or not last_count <= count <= len(native)
                or count and native[count-1]['copy']['second_copy_end'] > decision):
            raise ValueError('Causal online frame, sample count and host clocks required')
        bounds = [views[name]['render_started_clock']+low for name in names]
        if any(abs(x-y) > 1e-8 for x, y in zip(bounds, event['render_start_lower'], strict=True)):
            raise ValueError('Online view lower bounds differ from original prior clock exchange')
        save_lower = max(views[name]['save_finished_clock'] for name in names)+low
        if receipt < save_lower:
            raise ValueError('Host file receipt precedes earliest original save completion')
        # Candidate must belong to the snapshot of rows actually consumed online.
        consumed_time = native[count-1]['copy']['second_copy_end'] if count else -1.
        current_index = bisect_right(receipts, min(decision, consumed_time))-1
        if current_index <= 0:
            if event['status'] != 'missing_current_body':
                raise ValueError('Initial packet age is unknown')
            missing += 1
        else:
            if event['status'] != 'context_built':
                raise ValueError('Available current packet was discarded')
            choices = [bisect_left(receipts, min(lower, consumed_time+1e-9))-1 for lower in bounds]
            # Strict lower bound applies to render, while consumed_time is inclusive.
            choices = [min(choice, bisect_right(receipts, consumed_time)-1) for choice in choices]
            expected = [None if index <= 0 else packets[index] for index in choices]
            expected.append(packets[current_index])
            observed = [*event['view_body'], event['decision_body']]
            for j, (packet, data) in enumerate(zip(expected, observed, strict=True)):
                if packet is None:
                    if data is not None:
                        raise ValueError('Missing view history was silently filled')
                    continue
                if data is None or data['packet'] != packet['packet'] or data['receipt'] != packet['first_end']:
                    raise ValueError('Chosen original packet/first receipt differs')
                encoded = encoders[j].encode(packet['sample'], episode_id=result['request_id'],
                    decision_monotonic_seconds=receipt if j < 4 else decision)
                np.testing.assert_array_equal(encoded.valid, data['valid'])
                np.testing.assert_array_equal(encoded.values, data['values'])
                if j < 4 and encoded.valid[4]:
                    yaw += 1
            valid += 1
        build_ms.append(1000*(decision-receipt))
        camera_age_ms.append(1000*(receipt-save_lower))
        counts.append(count-last_count)
        last_decision, last_count = decision, count
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_sha256': sha256(root/'result.json'), 'clock_before_sha256_at_audit': sha256(root/'clock-before.json'),
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    output = {'status': 'completed_online_body_shadow_audit', 'events': len(events),
        'missing_current_groups': missing, 'context_groups': valid, 'valid_view_yaw': yaw,
        'receipt_to_selection_snapshot_ms_p50_p95_max': np.percentile(build_ms, [50, 95, 100]).tolist(),
        'save_lower_to_file_receipt_ms_p50_p95_max': np.percentile(camera_age_ms, [50, 95, 100]).tolist(),
        'maximum_native_copies_per_decision': max(counts),
        'clock_before_hash_originally_sealed': False,
        'physical_sample_or_exposure_time_verified': False, 'actual_neural_actor_executed': False,
        'training_admissible': False, 'optimizer_updates': 0}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
