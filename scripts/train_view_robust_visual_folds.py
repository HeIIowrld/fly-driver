"""Fit image-grid folds with TRAIN-only single-view mean interventions."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.runtime)
    plan = read(root/'20261030-t1692-fold-setup/manifest.json')['config']['plan']
    plan = {'dataset': str(Path(plan['source_data'])/'dataset.json'),
            'sensor_replay': str(Path(plan['sensor_replay'])/'result.json')}
    dataset = read(plan['dataset'])
    sensor = read(plan['sensor_replay'])
    names = ['episode-04-train', 'episode-05-train', 'episode-06-train']
    entries = {e['id']: e for e in dataset['episodes'] if e['role'] == 'train'}
    sources = {e['id']: e for e in sensor['episodes'] if e['role'] == 'train'}
    if sorted(entries) != names or sorted(sources) != names:
        raise ValueError('Exact original TRAIN histories required')
    run = begin_run(args.output, {'dataset_sha256': sha256(plan['dataset']),
        'sensor_replay_sha256': sha256(plan['sensor_replay']),
        'program_sha256': sha256(__file__), 'development_opened': False,
        'direct_or_registration_body_input': False, 'actor_deployed': False})
    records, pins = [], {}
    common_uv = None
    for name in names:
        entry, source = entries[name], sources[name]
        response = Path(source['output'])/'responses.npz'
        registration = Path(source['output'])/'retina-registration.npz'
        target = Path(entry['targets_file'])
        for path, expected in [(response, source['responses_sha256']),
                               (target, entry['targets_sha256'])]:
            if sha256(path) != expected:
                raise ValueError('Original response or target changed: ' + str(path))
        pins[str(response)] = sha256(response)
        pins[str(registration)] = sha256(registration)
        pins[str(target)] = sha256(target)
        with load_array(registration, allow_pickle=False) as data:
            uv = data['image_uv'].copy()
            if data['view_order'].tolist() != ['front', 'left', 'right', 'rear']:
                raise ValueError('Expected four recorded camera identities')
        if uv.shape != (721, 2) or not np.isfinite(uv).all():
            raise ValueError('Original fixed image registration required')
        if common_uv is None:
            common_uv = uv
        elif not np.array_equal(common_uv, uv):
            raise ValueError('Image-coordinate registration changed between histories')
        extent = np.ptp(uv, axis=0)
        if np.any(extent <= 0):
            raise ValueError('Both image dimensions must span nonzero extent')
        cell = np.minimum(((uv-uv.min(axis=0))/extent*4).astype(int), 3)
        weights = np.zeros((721, 16), dtype=np.float64)
        for index in range(16):
            selected = (cell[:, 0] == index % 4) & (cell[:, 1] == index // 4)
            if selected.any():
                weights[selected, index] = 1/selected.sum()
        with load_array(response, allow_pickle=False) as data:
            if data['additional_cell_types'].tolist() != ['L1', 'L2', 'Mi1', 'Tm3', 'Tm1', 'Tm9']:
                raise ValueError('Original retinal cell types required')
            if data['motion_activity'].shape != (739, 4, 8, 721):
                raise ValueError('Original four-view motion histories required')
            clock = data['decision_monotonic_seconds'].copy()
            retinal = np.asarray(data['additional_activity']) @ weights
            motion = np.asarray(data['motion_activity']) @ weights
        with load_array(target, allow_pickle=False) as data:
            np.testing.assert_array_equal(clock, data['observation_time'])
            y = data['preview'].reshape(-1, 6)[:, 5].copy()
            valid = data['preview_valid'][:, 2].copy()
        records.append({'id': name, 'retinal': retinal.reshape(739, -1),
            'motion': motion.reshape(739, -1), 'y': y, 'valid': valid})
        print(json.dumps({'loaded': name, 'nonempty_image_bins': int((weights.sum(0)>0).sum())}),
              flush=True)
    reports = []
    for held_index, held in enumerate(records):
        train = [e for i, e in enumerate(records) if i != held_index]
        for condition in ['both']:
            def features(record, condition=condition):
                if condition == 'both':
                    return np.concatenate([record['retinal'], record['motion']], axis=1)
                return record[condition]
            x = np.concatenate([features(e) for e in train])
            y = np.concatenate([e['y'] for e in train])
            valid = np.concatenate([e['valid'] for e in train])
            mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
            d = np.clip((x-mean)/scale, -5, 5)[valid]
            centre = float(y[valid].mean())
            gram = d.T @ d
            right = d.T @ (y[valid]-centre)
            for view in range(4):
                reduced = d.copy()
                reduced[:, view*96:(view+1)*96] = 0
                reduced[:, 384+view*128:384+(view+1)*128] = 0
                gram += reduced.T @ reduced
                right += reduced.T @ (y[valid]-centre)
            gram.flat[::len(gram)+1] += len(d)*5
            w = np.linalg.solve(gram, right)
            error = float(np.linalg.norm(gram @ w-right)/max(np.linalg.norm(right), 1e-12))
            if not np.isfinite(w).all() or error > 1e-8:
                raise ValueError('Ridge solve failed')
            full_pred = centre + np.clip((features(held)-mean)/scale, -5, 5) @ w
            np.savez_compressed(run/(held['id']+'-'+condition+'.npz'), mean=mean, scale=scale,
                                weight=w, centre=centre, prediction=full_pred)
            truth = held['y'][held['valid']]
            pred = full_pred[held['valid']]
            groups = {}
            for label, mask in [('negative', truth < -.001), ('positive', truth > .001)]:
                groups[label] = {'rows': int(mask.sum()), 'sign_accuracy': float(np.mean(
                    np.sign(pred[mask]) == np.sign(truth[mask]))) if mask.any() else None}
            reports.append({'artifact_sha256': sha256(run/(held['id']+'-'+condition+'.npz')),
                'held_out': held['id'], 'condition': condition,
                'rmse_per_m': float(np.sqrt(np.mean((pred-truth)**2))),
                'constant_train_mean_rmse_per_m': float(np.sqrt(np.mean((centre-truth)**2))),
                'valid_rows': int(len(truth)), 'direction_groups': groups,
                'ridge_relative_solve_error': error})
            print(json.dumps(reports[-1]), flush=True)
    if any(sha256(path) != expected for path, expected in pins.items()):
        raise ValueError('Original inputs changed during experiment')
    write_json(run/'result.json', {'status': 'completed_view_mean_augmented_train_folds',
        'reports': reports, 'source_hashes': pins, 'development_opened': False,
        'image_grid': '4x4 bins of fixed recorded image_uv in each of four views',
        'feature_inputs': 'saved author retinal and/or motion activity only',
        'body_input_or_tilt_registration': False, 'original_author_optimizer_updates': 0,
        'train_augmentation': 'original plus four single-view means, equal objective weight',
        'diagnostic_direction_threshold_per_m': .001, 'rows_are_not_independent': True,
        'target': '20m teacher-path chord curvature, not optimal line or corner distance',
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
