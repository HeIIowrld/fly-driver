"""Compare native controls over host receipt windows; never create training labels."""
import argparse
import base64
import hashlib
import json
import struct
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--body-audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    candidate = Path(a.candidate)
    recording = candidate/'recording'
    source = read(candidate/'result.json')
    audit = read(Path(a.body_audit)/'result.json')
    if (source['role'] != 'train' or source['fit_allowed'] is not False
            or audit['status'] != 'completed' or not audit['raw_values_masks_receipts_match']):
        raise ValueError('Preassigned fit-forbidden TRAIN and audited original body required')
    events_path, native_path = recording/'events.jsonl', recording/'native-copy-history.json'
    event_hash, native_hash = sha256(events_path),sha256(native_path)
    audited = read(Path(a.body_audit)/'manifest.json')['config']
    if audited['events_sha256'] != event_hash or audited['native_sha256'] != native_hash:
        raise ValueError('Audit must refer to same original event and native streams')
    run = begin_run(a.output, dict(program_sha256=sha256(__file__), events_sha256=event_hash,
        native_sha256=native_hash, body_audit_sha256=sha256(Path(a.body_audit)/'result.json'),
        optimizer_updates=0, fit_allowed=False, source_time_claim=False))
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    samples = read(native_path)
    times, controls, packets = [], [], []
    for row in samples:
        original = row['authority']
        raw = base64.b64decode(original['prefix_base64'],validate=True)
        if (len(raw) != 420 or hashlib.sha256(raw).hexdigest() != original['prefix_sha256']
                or struct.unpack_from('<i',raw,0)[0] != original['packet_id']):
            raise ValueError('Native copy bytes/hash/packet mismatch')
        times.append(float(original['received_monotonic_seconds']))
        packets.append(original['packet_id'])
        controls.append([struct.unpack_from('<f',raw,offset)[0] for offset in (24,4,8)])
    if np.any(np.diff(times) < 0) or not np.isfinite(controls).all():
        raise ValueError('Ordered finite original native receipts required')
    controls = np.asarray(controls)
    rows = []
    for index,event in enumerate(events):
        if event['index'] != index:
            raise ValueError('Original event ordering changed')
        start = min(float(v['render_started_lower']) for v in event['camera']['views'])
        end = float(event['decision_monotonic_seconds'])
        first = bisect_left(times,start)-1
        last = bisect_right(times,end)-1
        if first < 0 or last < first or start > end:
            raise ValueError('Missing original preceding native receipt or invalid host window')
        window = controls[first:last+1]
        rows.append(dict(index=index, host_window_seconds=end-start, native_copies=len(window),
            packet_changes=int(np.count_nonzero(np.diff(packets[first:last+1]))),
            native_steer_gas_brake_range=np.ptp(window,axis=0).tolist(),
            last_minus_first=(window[-1]-window[0]).tolist()))
    ranges = np.asarray([r['native_steer_gas_brake_range'] for r in rows])
    summary = dict(status='completed_receipt_window_diagnostic', role='train', observations=len(rows),
        rows_with_packet_changes=sum(r['packet_changes'] > 0 for r in rows),
        native_axis_range_median=np.median(ranges,axis=0).tolist(),
        native_axis_range_p95=np.percentile(ranges,95,axis=0).tolist(),
        native_axis_range_max=np.max(ranges,axis=0).tolist(),
        rows_with_axis_range_gt_0_01=(ranges > .01).sum(axis=0).tolist(),
        axis_order=['native_steerAngle','gas','brake'], fit_allowed=False,
        label_created=False, physical_exposure_or_sample_sync_verified=False,
        interpretation='Receipt-window variability only; raw steering is uncalibrated, '
                       'not a physical exposure interval or proof of control exclusivity.')
    write_json(run/'rows.json',rows)
    write_json(run/'result.json',summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
