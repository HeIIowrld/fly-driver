"""Independently verify TRAIN00 view priors and quantify partial availability."""

import base64
import hashlib
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root = base/'20261029-t1335-teacher-corpus-v2/episode-00-train'
    candidate, common = base/'20261031-train00-per-view-body', base/'20261031-historical-train00-temporal-body-context'
    cr, cm = read(candidate/'result.json'), read(candidate/'manifest.json')['config']
    if (sha256(root/'events.jsonl') != cm['events_sha256']
            or sha256(root/'control/native-motion-samples.json') != cm['native_samples_sha256']
            or sha256(candidate/'per-view-body.npz') != cr['arrays_sha256']):
        raise ValueError('Exact original and candidate sources required')
    events = [json.loads(s) for s in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    samples, times, hashes = [], [], {}
    for row in read(root/'control/native-motion-samples.json'):
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != row['prefix_sha256']:
            raise ValueError('Native payload changed')
        sample = decode_navigation_body_prefix(raw, row['received_monotonic_seconds'])
        packet = sample.legacy.packet_id
        if packet != row['packet_id'] or packet in hashes and hashes[packet] != row['prefix_sha256']:
            raise ValueError('Original packet identity or repeated bytes changed')
        if packet not in hashes:
            if times and sample.legacy.received_monotonic_seconds <= times[-1]:
                raise ValueError('Original first receipts must advance')
            times.append(sample.legacy.received_monotonic_seconds)
            samples.append(sample)
            hashes[packet] = row['prefix_sha256']
    encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                     native_identity=read(root/'identity.json')) for _ in VIEW_ORDER]
    with np.load(candidate/'per-view-body.npz', allow_pickle=False) as f, np.load(common/'body-pairs.npz', allow_pickle=False) as c:
        arrays = {k:f[k].copy() for k in ('values','valid','packets','receipts','render_start_lower','decisions')}
        common_packets = c['packets'][:, 1].copy()
        original_decisions = c['decision_monotonic_seconds'].copy()
    if len(events) != 741 or not np.array_equal(arrays['decisions'], original_decisions):
        raise ValueError('All original decisions required')
    reason_counts = {'matched':0, 'no_prior':0, 'too_old':0}
    for i,event in enumerate(events):
        views = {v['view_id']:v for v in event['camera']['views']}
        camera = event['camera']['received']
        for j,name in enumerate(VIEW_ORDER):
            lower = views[name]['render_started_lower']
            if arrays['render_start_lower'][i,j] != lower:
                raise ValueError('Original view render bound changed')
            index = bisect_left(times, lower)-1
            present = index >= 0 and 0 <= camera-times[index] <= .05
            reason_counts['matched' if present else ('no_prior' if index < 0 else 'too_old')] += 1
            if not present:
                if arrays['packets'][i,j] != -1 or arrays['valid'][i,j].any():
                    raise ValueError('Missing prior was imputed')
                continue
            obs = encoders[j].encode(samples[index], episode_id='independent-original', decision_monotonic_seconds=camera)
            if (arrays['packets'][i,j] != obs.packet_id or arrays['receipts'][i,j] != obs.original_received_monotonic_seconds
                    or not np.array_equal(arrays['valid'][i,j], obs.valid)
                    or not np.array_equal(arrays['values'][i,j], obs.values)):
                raise ValueError('Native view body differs')
    present = arrays['packets'] >= 0
    extras = (present & (common_packets[:,None] < 0)).sum(axis=0).tolist()
    distinct = sum(len(set(row[row >= 0])) > 1 for row in arrays['packets'])
    partial = int((present.any(axis=1) & ~present.all(axis=1)).sum())
    run = begin_run(base/'20261031-train00-per-view-independent-audit',
        {'program_sha256':sha256(__file__), 'candidate_sha256':cr['arrays_sha256'],
         'common_sha256':sha256(common/'body-pairs.npz'), 'events_sha256':cm['events_sha256'],
         'scope':'host-first-receipt ordering only, never source physics/exposure'})
    output = {'status':'completed', 'observations':len(events), 'view_reasons':reason_counts,
              'valid_where_common_missing_per_view':extras, 'groups_with_distinct_present_packets':distinct,
              'partially_available_view_groups':partial, 'raw_native_values_masks_receipts_match':True,
              'training_admissible':False, 'physical_timing_verified':False, 'vehicle_control_sent':False}
    write_json(run/'result.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
