"""Diagnose full new-mount histories: feature shift, turn coverage and unfitted body kinematics."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.data)
    manifest = read(root/'dataset.json')
    episodes = []
    for entry in manifest['episodes']:
        if entry['role'] not in ('train', 'development'):
            raise ValueError('No evaluation access')
        for key in ('features', 'targets'):
            if sha256(entry[key+'_file']) != entry[key+'_sha256']:
                raise ValueError('Original data changed')
        body = Path(entry['body_replay'])
        proof = read(body/'result.json')
        if (sha256(body/'result.json') != entry['body_result_sha256']
                or sha256(body/'virtual-body.npz') != proof['virtual_body_sha256']
                or not proof['legacy16_whole_replay_exact']):
            raise ValueError('Whole unchanged body21 required')
        with load_array(entry['features_file'], allow_pickle=False) as f, load_array(
                body/'virtual-body.npz', allow_pickle=False) as b:
            np.testing.assert_array_equal(f['observation_time'], b['decision_monotonic_seconds'])
            episodes.append(dict(entry, x={k: f[k].copy() for k in (
                'raw_control', 'processed_actual')}, y=f['preview'].copy(),
                valid=f['preview_valid'].copy(), body=b['values'].copy(),
                body_valid=b['valid'].copy()))
    moments = {}
    for key in ('raw_control', 'processed_actual'):
        x = np.concatenate([e['x'][key] for e in episodes if e['role'] == 'train'])
        moments[key] = x.mean(0, dtype=float), x.astype(float).std(0).clip(1e-5)
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        dataset_sha256=sha256(root/'dataset.json'), no_fitting=True,
        kinematic_hypothesis='constant horizontal curvature: bearing=-yaw_left*s/(2*speed); opposite sign reported without selecting a winner',
        limitation='turn transitions, slip, exposure/body timing and sensory registration prevent treating this as a future-path oracle'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports = []
    for e in episodes:
        shifts = {}
        for key, (mean, scale) in moments.items():
            z = ((e['x'][key]-mean)/scale).reshape(-1, 4, 14, 16)
            shifts[key] = dict(fraction_over5_by_view=(np.abs(z) > 5).mean(
                axis=(0, 2, 3)).tolist(), mean_abs_by_view=np.abs(z).mean(axis=(0, 2, 3)).tolist())
        targets = []
        body_ok = e['body_valid'][:, 4] & e['body_valid'][:, 16] & (e['body'][:, 16] > 1)
        for h, distance in enumerate((5., 10., 20.)):
            y = e['y'][:, h, 0]
            valid = e['valid'][:, h]
            keep = valid & body_ok
            estimate = -e['body'][keep, 4]*distance/(2*e['body'][keep, 16])
            actual = y[keep]
            targets.append(dict(distance_m=distance, valid=int(valid.sum()),
                bearing_range_degrees=np.degrees([y[valid].min(), y[valid].max()]).tolist(),
                bearing_mean_degrees=float(np.degrees(y[valid].mean())),
                negative_left=int((y[valid] < -.01).sum()), neutral=int((np.abs(y[valid]) <= .01).sum()),
                positive_right=int((y[valid] > .01).sum()), body_valid=int(keep.sum()),
                registered_sign_kinematic_rmse=float(np.sqrt(np.mean((estimate-actual)**2))),
                opposite_sign_kinematic_rmse=float(np.sqrt(np.mean((-estimate-actual)**2))),
                zero_rmse_on_same_body_rows=float(np.sqrt(np.mean(actual**2)))))
        reports.append(dict(id=e['id'], role=e['role'], feature_shift=shifts, targets=targets))
    write_json(run/'result.json', dict(status='completed_exterior_preview_diagnosis',
        reports=reports, optimizer_updates=0, development_used_to_fit=False,
        motor_or_visual_registration_claim=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
