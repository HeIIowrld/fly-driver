"""Known-data calibration test: do original motion responses encode measured body yaw?"""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def describe(prediction, target, selected):
    if not selected.any():
        return dict(count=0)
    p, y = prediction[selected], target[selected]
    variance = float(np.square(y-y.mean()).sum())
    return dict(count=len(y), rmse=float(np.sqrt(np.mean((p-y)**2))),
        prediction_mean=float(p.mean()), target_mean=float(y.mean()),
        correlation=float(np.corrcoef(p, y)[0, 1]) if min(p.std(), y.std()) > 1e-10 else None,
        r2=1-float(np.square(p-y).sum())/variance if variance > 0 else None,
        sign_fraction=float((np.sign(p) == np.sign(y)).mean()))


def past_indices(times, lag, tolerance):
    query = times-lag
    right = np.searchsorted(times, query).clip(0, len(times)-1)
    left = (right-1).clip(0, len(times)-1)
    selected = np.where(np.abs(times[right]-query) < np.abs(times[left]-query), right, left)
    valid = ((query >= times[0]) & (np.abs(times[selected]-query) <= tolerance)
             & (times[selected] <= times))
    return selected, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan['data'])
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
        script_sha256=sha256(__file__), plan_sha256=sha256(args.plan),
        dataset_sha256=sha256(root/'dataset.json'), camera_sha256=sha256(plan['camera_config']),
        body_as_target_only=True, optimizer_updates=0, actual_vehicle_requests=0))
    camera = read(plan['camera_config'])
    episodes, registration_hashes = [], {}
    for entry in read(root/'dataset.json')['episodes']:
        assert entry['role'] in ('train', 'development')
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        visual = Path(entry['original']['scene_replay'])
        assert sha256(visual/'responses.npz') == entry['source_visual_sha256']
        registration = visual/'retina-registration.npz'
        registration_hashes[entry['file']] = sha256(registration)
        with load_array(registration, allow_pickle=False) as archive:
            pool = VisualSelfMotionPool.from_registration(camera, archive)
        with load_array(visual/'responses.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(a['observation_time'],
                                          archive['decision_monotonic_seconds'])
            motion = archive['motion_activity'].copy()
        whole = pool.features(motion)
        separate = []
        for view in range(4):
            retained = np.zeros_like(motion)
            retained[:, view] = motion[:, view]
            separate.append(pool.features(retained))
        assert np.max(np.abs(sum(separate)-whole)) < 1e-10
        features = dict(geometry_opponent6=whole,
            per_camera_geometry24=np.concatenate(separate, axis=1),
            coarse_motion512=a['scene_features'].reshape(-1, 4, 9, 16)[:, :, 1:].reshape(len(a[
                'observation_time']), -1))
        episodes.append((entry, a, features))
    reports, predictions, models = {}, {}, {}
    for lag in plan['target_lags_seconds']:
        targets = []
        for _, a, _ in episodes:
            index, valid = past_indices(a['observation_time'], lag,
                                        plan['past_target_match_tolerance_seconds'])
            valid &= a['body_valid'][index, 4]
            targets.append((a['body'][index, 4].astype(float), valid, index))
        training_y = np.concatenate([y[v] for (e, _, _), (y, v, _) in
            zip(episodes, targets, strict=True) if e['role'] == 'train'])
        train_mean = float(training_y.mean())
        for variant in plan['variants']:
            train_frames = np.concatenate([f[variant] for e, _, f in episodes
                                           if e['role'] == 'train'])
            mean = train_frames.mean(0)
            scale = np.maximum(train_frames.std(0), plan['normalization_std_floor'])
            normalized = [np.clip((f[variant]-mean)/scale, -plan['feature_clip'],
                                 plan['feature_clip']) for _, _, f in episodes]
            x = np.concatenate([z[v] for z, (e, _, _), (_, v, _) in
                zip(normalized, episodes, targets, strict=True) if e['role'] == 'train'])
            x = np.column_stack([np.ones(len(x)), x])
            penalty = np.eye(x.shape[1])*plan['ridge_penalty']
            penalty[0, 0] = 0
            coefficient = np.linalg.solve(x.T@x+penalty, x.T@training_y)
            key = variant+'_lag'+str(lag)
            models[key+'_coefficient'], models[key+'_mean'], models[key+'_scale'] = (
                coefficient, mean, scale)
            reports[key] = []
            for z, (entry, _a, _), (y, valid, index) in zip(
                    normalized, episodes, targets, strict=True):
                estimate = np.column_stack([np.ones(len(z)), z])@coefficient
                active = valid & (np.abs(y) >= plan['active_yaw_threshold_rad_s'])
                baseline = np.full(len(y), train_mean)
                reports[key].append(dict(file=entry['file'], role=entry['role'],
                    all=describe(estimate, y, valid), active=describe(estimate, y, active),
                    train_mean=describe(baseline, y, valid),
                    zero=describe(np.zeros(len(y)), y, valid)))
                predictions[key+'_'+entry['file'].removesuffix('.npz')] = estimate
                predictions['matched_'+str(lag)+'_'+entry['file'].removesuffix('.npz')] = index
        write_json(run/'progress.json', dict(completed=list(reports)))
    np.savez_compressed(run/'diagnostic-models.npz', **models)
    np.savez_compressed(run/'diagnostic-predictions.npz', **predictions)
    write_json(run/'result.json', dict(
        status='completed_visual_body_yaw_diagnosis', metrics=reports,
        checked_registration_sha256=registration_hashes,
        neuron_type_identity_independently_verified=False,
        source_arrays_or_clocks_modified=False, body_input_to_predictors=False,
        existing_stationary_reference_guard_bypassed=False,
        evaluation_role_arrays_opened=False, new_fly_network_weights_trained=False,
        actor_qualified=False, physiological_calibration_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
