"""Independently check pace counterfactual outputs against original native bytes."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, replay = Path(a.recording), Path(a.replay)
    result, manifest = read(replay/'result.json'), read(replay/'manifest.json')['config']
    if (result.get('status') != 'completed_pace_visual_body_counterfactual'
            or result.get('training_admissible') is not False
            or manifest['events_sha256'] != sha256(root/'events.jsonl')
            or sha256(replay/'events.json') != result['events_sha256']):
        raise ValueError('Exact original diagnostic and completed replay required')
    source = [json.loads(x) for x in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    output = read(replay/'events.json')
    if len(source) != len(output) or len(source) != 731:
        raise ValueError('Complete original pace history required')
    moving = []
    for i, (event, row) in enumerate(zip(source, output, strict=True)):
        authority = event['native']
        raw = base64.b64decode(authority['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != authority['prefix_sha256']:
            raise ValueError('Original native bytes differ')
        native = decode_physics_prefix(raw, authority['received_monotonic_seconds'])
        brake = native.fields['brake'] > .05 and native.fields['speedKmh'] > 1
        if (row['index'] != i or row['packet'] != native.packet_id
                or row['time'] != event['decision_monotonic_seconds']
                or row['moving_brake'] != brake):
            raise ValueError('Original packet, clock or moving-brake classification differs')
        moving.append(brake)
    values = np.asarray([row['intents'] for row in output])
    moving = np.asarray(moving, dtype=bool)
    if values.shape != (731, 3, 2) or not np.isfinite(values).all() or moving.sum() != 62:
        raise ValueError('All finite original outputs and 62 moving-brake decisions required')
    conditions = {}
    for index, name in ((1, 'tilt_zero'), (2, 'load_zero')):
        delta = np.abs(values[:, index]-values[:, 0])
        conditions[name] = {'mean_abs': delta.mean(axis=0).tolist(),
                            'moving_brake_mean_abs': delta[moving].mean(axis=0).tolist(),
                            'moving_brake_max_abs': delta[moving].max(axis=0).tolist()}
        if not np.allclose(conditions[name]['moving_brake_mean_abs'],
                           result['interventions'][name]['braking_mean_abs'], atol=1e-12, rtol=0):
            raise ValueError('Saved summary differs from original intervention events')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'source_events_sha256': sha256(root/'events.jsonl'),
        'replay_result_sha256': sha256(replay/'result.json'), 'seed': None})
    summary = {'status': 'completed_pace_visual_body_counterfactual_audit',
        'original_native_packets_verified': len(source), 'moving_brake_count': int(moving.sum()),
        'conditions': conditions, 'camera_mount_domain_shift': True,
        'fixed_goal_not_original_pace_task': True, 'fit_or_control': False,
        'physical_exposure_time_verified': False, 'physiological_benefit_verified': False}
    write_json(run/'result.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
