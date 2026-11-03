"""Vectorized independent audit of train-view receipt coverage, without labels."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    dataset = base/'20261029-t1434-body21-flight-cache/dataset.json'
    survey = base/'20261031-train-six-per-view-yaw-survey/result.json'
    report = read(survey)
    if report['status'] != 'completed':
        raise ValueError('Completed original coverage survey required')
    by_name = {e['episode']:e for e in report['episodes']}
    output = []
    for entry in read(dataset)['episodes']:
        if entry['role'] != 'train':
            continue
        row = by_name[entry['file']]
        root = Path(entry['original']['recording'])
        events_path, raw_path = root/'events.jsonl', root/'control/native-motion-samples.json'
        if sha256(events_path) != row['events_sha256'] or sha256(raw_path) != row['native_samples_sha256']:
            raise ValueError('Original survey source changed')
        samples = read(raw_path)
        receipts, packets, seen = [], [], {}
        for x in samples:
            raw = base64.b64decode(x['prefix_base64'], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            packet = decode_physics_prefix(raw, x['received_monotonic_seconds']).packet_id
            if digest != x['prefix_sha256'] or packet != x['packet_id']:
                raise ValueError('Original native payload mismatch')
            if packet in seen:
                if seen[packet] != digest:
                    raise ValueError('Repeated packet changed bytes')
                continue
            seen[packet] = digest
            receipts.append(x['received_monotonic_seconds']); packets.append(packet)
        receipts, packets = np.asarray(receipts), np.asarray(packets)
        if not np.all(np.diff(receipts) > 0):
            raise ValueError('First receipt times must advance')
        events = [json.loads(s) for s in events_path.read_text(encoding='utf8').splitlines()]
        starts, received = [], []
        for event in events:
            view = {x['view_id']:x for x in event['camera']['views']}
            starts.append([view[name]['render_started_lower'] for name in VIEW_ORDER])
            received.append(event['camera']['received'])
        starts, received = np.asarray(starts), np.asarray(received)
        index = np.searchsorted(receipts, starts, side='left')-1
        safe = np.maximum(index, 0)
        valid = (index >= 0) & (received[:,None]-receipts[safe] >= 0) & (received[:,None]-receipts[safe] <= .05)
        chosen = np.where(valid, packets[safe], -1)
        distinct = sum(len(set(x[x >= 0])) > 1 for x in chosen)
        partial = int((valid.any(axis=1) & ~valid.all(axis=1)).sum())
        if (valid.sum(axis=0).tolist() != row['matched_views'] or distinct != row['distinct_packet_groups']
                or partial != row['partially_available_groups']):
            raise ValueError('Independent vectorized coverage differs')
        output.append({'episode':entry['file'],'observations':len(events),
            'matched_views':valid.sum(axis=0).tolist(),'distinct_packet_groups':int(distinct),
            'partially_available_groups':partial,
            'camera_view_span_ms_p50_p95_max':np.percentile((starts.max(axis=1)-starts.min(axis=1))*1000,[50,95,100]).tolist(),
            'native_first_receipt_gap_ms_p50_p95_max':np.percentile(np.diff(receipts)*1000,[50,95,100]).tolist()})
    if len(output) != 6:
        raise ValueError('All six TRAIN sources required')
    run = begin_run(base/'20261031-train-six-view-yaw-independent-audit',
        {'program_sha256':sha256(__file__), 'dataset_sha256':sha256(dataset), 'survey_sha256':sha256(survey),
         'scope':'independent receipt vectorization, no label/fit or physical exposure'})
    write_json(run/'result.json', {'status':'completed','episodes':output,
        'distinct_packet_groups':sum(x['distinct_packet_groups'] for x in output),
        'training_admissible':False,'vehicle_control_sent':False})
    print(json.dumps(read(run/'result.json')))


if __name__ == '__main__':
    main()
