"""Independently check per-view selection and values against saved raw bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'derived', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root, derived = Path(a.recording), Path(a.derived)
    report, config = read(derived/'result.json'), read(derived/'manifest.json')['config']
    ep, cp, arrays = root/'events.jsonl', root/'native-copy-history.json', derived/'per-view-body.npz'
    if (report['status'] != 'completed_high_rate_per_view_body'
            or report['training_admissible'] is not False
            or sha256(arrays) != report['arrays_sha256']
            or sha256(ep) != config['events_sha256'] or sha256(cp) != config['native_sha256']):
        raise ValueError('Exact original and derived data required')
    samples, receipts, seen = [], [], {}
    previous = -1.
    for row in read(cp):
        auth, timing = row['authority'], row['copy']
        raw = base64.b64decode(auth['prefix_base64'], validate=True)
        end = timing['second_copy_end']
        if (hashlib.sha256(raw).hexdigest() != auth['prefix_sha256']
                or not previous <= timing['first_copy_start'] <= end
                or end != auth['received_monotonic_seconds']):
            raise ValueError('Raw native bytes or host order differ')
        previous = end
        sample = decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size], end)
        packet = sample.legacy.packet_id
        if packet != auth['packet_id'] or packet in seen and seen[packet] != auth['prefix_sha256']:
            raise ValueError('Native packet identity or repeat differs')
        if packet not in seen:
            seen[packet] = auth['prefix_sha256']
            samples.append(sample)
            receipts.append(end)
    events = [json.loads(line) for line in ep.read_text(encoding='utf8').splitlines()]
    if len(events) != report['decisions'] or np.any(np.diff(receipts) <= 0):
        raise ValueError('Original decision and receipt order required')
    encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                native_identity=read(root/'identity.json')) for _ in range(5)]
    with np.load(arrays, allow_pickle=False) as f:
        x = {key:f[key].copy() for key in f.files}
    matched = missing = 0
    for i, event in enumerate(events):
        camera = event['camera']
        views = {v['view_id']:v for v in camera['views']}
        body = event['native']
        raw = base64.b64decode(body['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != body['prefix_sha256']:
            raise ValueError('Current native hash differs')
        sample = decode_navigation_body_prefix(raw, body['received_monotonic_seconds'])
        current = encoders[4].encode(sample, episode_id='independent',
                                     decision_monotonic_seconds=event['decision_monotonic_seconds'])
        if (event['index'] != i or x['decisions'][i] != event['decision_monotonic_seconds']
                or not np.array_equal(x['current_values'][i],current.values)
                or not np.array_equal(x['current_valid'][i],current.valid)):
            raise ValueError('Current body or decision differs')
        for j, view in enumerate(VIEW_ORDER):
            lo = views[view]['render_started_lower']
            index = int(np.searchsorted(receipts, lo, side='left'))-1
            present = index >= 0 and 0 <= camera['received']-receipts[index] <= .05
            if x['render_start_lower'][i,j] != lo:
                raise ValueError('Original render bound differs')
            if not present:
                missing += 1
                if x['packets'][i,j] != -1 or x['receipts'][i,j] != -1 or x['valid'][i,j].any():
                    raise ValueError('Missing context was filled')
                continue
            obs = encoders[j].encode(samples[index], episode_id='independent',
                                     decision_monotonic_seconds=camera['received'])
            if (x['packets'][i,j] != obs.packet_id or x['receipts'][i,j] != receipts[index]
                    or not np.array_equal(x['valid'][i,j],obs.valid)
                    or not np.array_equal(x['values'][i,j],obs.values)):
                raise ValueError('Reconstructed per-view native differs')
            matched += 1
    run = begin_run(a.output, dict(program_sha256=sha256(__file__), events_sha256=sha256(ep),
        native_sha256=sha256(cp), arrays_sha256=sha256(arrays), scope=__doc__))
    out = dict(status='completed', decisions=len(events), matched_views=matched, missing_views=missing,
        raw_values_masks_receipts_match=True, training_admissible=False,
        physical_timing_verified=False, optimizer_updates=0)
    write_json(run/'result.json',out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
