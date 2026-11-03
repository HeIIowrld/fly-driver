"""Reconstruct recorded per-view body contexts from original native copies."""

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.per_view_body_history import PerViewBodyHistory
from scripts.audit_causal_copy_body_context import first_observed_packets


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def compare(saved, reconstructed):
    if saved is None or reconstructed is None:
        return saved is None and reconstructed is None
    return (saved['packet'] == reconstructed.packet_id
            and saved['receipt'] == reconstructed.original_received_monotonic_seconds
            and np.array_equal(np.asarray(saved['values']), reconstructed.values)
            and np.array_equal(np.asarray(saved['valid'], bool), reconstructed.valid))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root = base/'20261031-process-body-owned-pause-live'
    result = read(root/'result.json')
    if (sha256(root/'bridge-events.json') != result['bridge_sha256']
            or sha256(root/'native-copy-windows.json') != result['native_sha256']):
        raise ValueError('Original native and visual sources required')
    bridge, native = read(root/'bridge-events.json'), read(root/'native-copy-windows.json')
    packets = first_observed_packets(native)
    receipt_times = [x['first_end'] for x in packets]
    history = PerViewBodyHistory(episode_id=result['camera_request_id'],
        native_identity=read('configs/ac-virtual-body-v1.json')['native_identity'])
    pointer, total, missing_context, missing_views = 0, 0, 0, 0
    distinct_valid_groups, yaw_spreads = 0, []
    for event in bridge:
        status = event.get('neural', {}).get('status')
        if status not in ('visual_published','missing_current_body'):
            continue
        visual = event['visual']
        count = visual['native_rows_consumed']
        if not 0 < count <= len(native):
            raise ValueError('Original native drain count required')
        last_end = native[count-1]['copy']['second_copy_end']
        limit = bisect_right(receipt_times, last_end)
        while pointer < limit:
            history.ingest(packets[pointer]['sample'])
            pointer += 1
        actual = history.build(render_start_lowers=tuple(visual['render_start_lower']),
            camera_receipt_time=visual['camera_receipt'], decision_time=visual['decision'])
        total += 1
        if actual is None:
            if status != 'missing_current_body':
                raise ValueError('Missing current body was not preserved')
            missing_context += 1
            continue
        if status != 'visual_published' or not compare(visual['decision_body'], actual.decision_body):
            raise ValueError('Current recorded body differs from native reconstruction')
        for expected, reconstructed in zip(visual['view_body'], actual.visual_contexts, strict=True):
            if not compare(expected, reconstructed):
                raise ValueError('Per-view prior differs from native reconstruction')
            missing_views += expected is None
        usable = [x for x in actual.visual_contexts if x is not None and x.valid[4]]
        distinct_valid_groups += len({x.packet_id for x in usable}) > 1
        if len(usable) > 1:
            yaws = [float(x.values[4]) for x in usable]
            yaw_spreads.append(max(yaws)-min(yaws))
    run = begin_run(base/'20261031-process-shadow-visual-context-raw-audit-v2',
        {'program_sha256':sha256(__file__), 'recording_sha256':sha256(root/'result.json'),
         'native_sha256':result['native_sha256'], 'bridge_sha256':result['bridge_sha256'],
         'scope':'original host receipt ordering and encoded body, no physical timing claim'})
    write_json(run/'result.json', {'status':'completed', 'processed_visual_groups':total,
        'groups_without_current_body':missing_context, 'published_groups':total-missing_context,
        'missing_per_view_prior_count':missing_views, 'native_values_masks_and_receipts_match':True,
        'distinct_valid_yaw_packet_groups':int(distinct_valid_groups),
        'within_group_yaw_spread_p50_p95_max':np.percentile(yaw_spreads, [50,95,100]).tolist(),
        'physical_sample_or_exposure_verified':False,'vehicle_control_sent':False})
    print(json.dumps(read(run/'result.json')))


if __name__ == '__main__':
    main()
