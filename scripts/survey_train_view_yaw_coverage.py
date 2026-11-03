"""Survey train-role original per-view body receipts; never fit or open labels."""

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
    dataset = base/'20261029-t1434-body21-flight-cache/dataset.json'
    entries = [x for x in read(dataset)['episodes'] if x['role'] == 'train']
    if len(entries) != 6 or len({x['file'] for x in entries}) != 6:
        raise ValueError('Six unchanged original TRAIN roles required')
    summaries = []
    for entry in entries:
        root = Path(entry['original']['recording'])
        events_path, native_path = root/'events.jsonl', root/'control/native-motion-samples.json'
        if sha256(events_path) != entry['source_events_sha256'] or entry['original']['role'] != 'train':
            raise ValueError('Exact original TRAIN event source required')
        events = [json.loads(x) for x in events_path.read_text(encoding='utf8').splitlines()]
        if len(events) != entry['observations']:
            raise ValueError('Complete original episode required')
        samples, times, seen, previous_receipt = [], [], {}, -1.
        for row in read(native_path):
            raw = base64.b64decode(row['prefix_base64'], validate=True)
            stamp = row['received_monotonic_seconds']
            if hashlib.sha256(raw).hexdigest() != row['prefix_sha256'] or stamp <= previous_receipt:
                raise ValueError('Original native hashes and receipt order required')
            sample = decode_navigation_body_prefix(raw, stamp)
            packet = sample.legacy.packet_id
            if packet != row['packet_id'] or samples and packet < samples[-1].legacy.packet_id:
                raise ValueError('Original packet order required')
            if packet in seen and seen[packet] != row['prefix_sha256']:
                raise ValueError('Duplicate packet bytes changed')
            if packet not in seen:
                samples.append(sample); times.append(stamp); seen[packet] = row['prefix_sha256']
            previous_receipt = stamp
        encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                    native_identity=read(root/'identity.json')) for _ in VIEW_ORDER]
        valid_counts = np.zeros(4, dtype=np.int64)
        distinct_groups = partial_groups = 0
        yaw_spreads, yaw_magnitudes = [], []
        for i,event in enumerate(events):
            views = {v['view_id']:v for v in event['camera']['views']}
            if event['index'] != i or set(views) != set(VIEW_ORDER):
                raise ValueError('Four ordered original views required')
            received = event['camera']['received']
            selected, yaws = [], []
            for j,name in enumerate(VIEW_ORDER):
                k = bisect_left(times, views[name]['render_started_lower'])-1
                if k < 0 or not 0 <= received-times[k] <= .05:
                    continue
                obs = encoders[j].encode(samples[k], episode_id=entry['file'], decision_monotonic_seconds=received)
                valid_counts[j] += 1
                selected.append(obs.packet_id)
                if obs.valid[4]:
                    yaws.append(float(obs.values[4]))
                    yaw_magnitudes.append(abs(float(obs.values[4])))
            distinct_groups += len(set(selected)) > 1
            partial_groups += 0 < len(selected) < 4
            if len(yaws) >= 2:
                yaw_spreads.append(max(yaws)-min(yaws))
        summaries.append({'episode':entry['file'],'role':'train','observations':len(events),
            'events_sha256':entry['source_events_sha256'],'native_samples_sha256':sha256(native_path),
            'matched_views':valid_counts.tolist(),'distinct_packet_groups':int(distinct_groups),
            'partially_available_groups':int(partial_groups),
            'within_group_yaw_spread_p50_p95_max':np.percentile(yaw_spreads,[50,95,100]).tolist() if yaw_spreads else None,
            'valid_yaw_abs_p50_p95_max':np.percentile(yaw_magnitudes,[50,95,100]).tolist() if yaw_magnitudes else None})
    run = begin_run(base/'20261031-train-six-per-view-yaw-survey',
        {'program_sha256':sha256(__file__), 'dataset_sha256':sha256(dataset),
         'scope':'TRAIN-role source receipt coverage only; no arrays, labels, fit or deployment'})
    output = {'status':'completed','episodes':summaries,
        'total_distinct_packet_groups':sum(x['distinct_packet_groups'] for x in summaries),
        'training_admissible':False,'physical_sample_or_exposure_verified':False,'vehicle_control_sent':False}
    write_json(run/'result.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
