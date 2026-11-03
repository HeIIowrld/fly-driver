"""Reconstruct original per-view body contexts from first native copy receipts."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.causal_render_body_match import select_prior_body_receipts_per_view
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix
from scripts.audit_causal_copy_body_context import first_observed_packets


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'raw-audit', 'native-audit', 'output'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    root, ra, na = map(Path, (a.recording, a.raw_audit, a.native_audit))
    result, raw, native = read(root/'result.json'), read(ra/'result.json'), read(na/'result.json')
    rc, nc = read(ra/'manifest.json')['config'], read(na/'manifest.json')['config']
    ep, cp = root/'events.jsonl', root/'native-copy-history.json'
    if (result['status'] != 'completed' or result['training_admissible'] is not False
            or raw['status'] != 'completed' or native['status'] != 'completed'
            or any(raw['actor_replay_max_errors'].values()) or not raw['camera_aid_state_verified']
            or rc['events_sha256'] != sha256(ep) or nc['events_sha256'] != sha256(ep)
            or nc['native_sha256'] != sha256(cp)):
        raise ValueError('Matching complete original camera and native audits required')
    events = [json.loads(line) for line in ep.read_text(encoding='utf8').splitlines()]
    packets = first_observed_packets(read(cp))
    receipts = [packet['first_end'] for packet in packets]
    identity = read(root/'identity.json')
    contract = 'configs/ac-virtual-navigation-body-v2.json'
    encoders = [NavigationBodyEncoder(contract, native_identity=identity) for _ in range(5)]
    n = len(events)
    if n != raw['decisions'] or n != native['events']:
        raise ValueError('All original events required')
    values = np.zeros((n, 4, 21), dtype=np.float64)
    valid = np.zeros((n, 4, 21), dtype=bool)
    ids = np.full((n, 4), -1, dtype=np.int64)
    times = np.full((n, 4), -1., dtype=np.float64)
    current_values = np.zeros((n, 21), dtype=np.float64)
    current_valid = np.zeros((n, 21), dtype=bool)
    lower = np.zeros((n, 4), dtype=np.float64)
    decision = np.zeros(n, dtype=np.float64)
    reasons = []
    for i, event in enumerate(events):
        if event['index'] != i:
            raise ValueError('Original complete event order required')
        camera, body = event['camera'], event['native']
        views = {v['view_id']: v for v in camera['views']}
        if set(views) != set(VIEW_ORDER):
            raise ValueError('Original four views required')
        raw_body = base64.b64decode(body['prefix_base64'], validate=True)
        if hashlib.sha256(raw_body).hexdigest() != body['prefix_sha256']:
            raise ValueError('Original body bytes differ')
        sample = decode_navigation_body_prefix(raw_body, body['received_monotonic_seconds'])
        if sample.legacy.packet_id != body['packet_id']:
            raise ValueError('Original body packet differs')
        decision[i] = event['decision_monotonic_seconds']
        current = encoders[4].encode(sample, episode_id='candidate',
                                      decision_monotonic_seconds=decision[i])
        current_values[i], current_valid[i] = current.values, current.valid
        lower[i] = [views[v]['render_started_lower'] for v in VIEW_ORDER]
        matches = select_prior_body_receipts_per_view(receipts, lower[i], camera['received'])
        reasons.append([m.reason for m in matches])
        for j, match in enumerate(matches):
            if match.native_index is None:
                continue
            observation = encoders[j].encode(packets[match.native_index]['sample'],
                episode_id='candidate', decision_monotonic_seconds=camera['received'])
            values[i,j], valid[i,j] = observation.values, observation.valid
            ids[i,j], times[i,j] = observation.packet_id, observation.original_received_monotonic_seconds
    if np.any(np.diff(decision) <= 0):
        raise ValueError('Original decisions must advance')
    run = begin_run(a.output, dict(program_sha256=sha256(__file__), events_sha256=sha256(ep),
        native_sha256=sha256(cp), raw_audit_sha256=sha256(ra/'result.json'),
        native_audit_sha256=sha256(na/'result.json'), contract_sha256=sha256(contract),
        scope='original host receipt order only; no fit permission'))
    np.savez_compressed(run/'per-view-body.npz', values=values, valid=valid, packets=ids,
        receipts=times, render_start_lower=lower, decisions=decision,
        current_values=current_values, current_valid=current_valid)
    write_json(run/'selection-reasons.json', reasons)
    out = dict(status='completed_high_rate_per_view_body', decisions=n,
        present_per_view=(ids >= 0).sum(axis=0).tolist(),
        distinct_prior_groups=sum(len(set(row[row >= 0])) > 1 for row in ids),
        missing_views=int((ids < 0).sum()), arrays_sha256=sha256(run/'per-view-body.npz'),
        reasons_sha256=sha256(run/'selection-reasons.json'),
        training_admissible=False, physical_timing_verified=False, optimizer_updates=0)
    write_json(run/'result.json', out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
