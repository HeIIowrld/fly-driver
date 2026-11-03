"""Intervene on individual fly-vision views in fixed-image-coordinate TRAIN folds."""

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
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Original replay plan changed')
    plan = read(args.plan)
    dataset = read(plan['dataset'])
    sensor = read(plan['sensor_replay'])
    names = ['episode-04-train', 'episode-05-train', 'episode-06-train']
    entries = {e['id']: e for e in dataset['episodes'] if e['role'] == 'train'}
    sources = {e['id']: e for e in sensor['episodes'] if e['role'] == 'train'}
    if sorted(entries) != names or sorted(sources) != names:
        raise ValueError('Exact original TRAIN histories required')
    run = begin_run(args.output, {'original_plan_sha256': args.plan_sha256,
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
            'motion': motion.reshape(739, -1), 'y': y, 'valid': valid, 'clock': clock})
        print(json.dumps({'loaded': name, 'nonempty_image_bins': int((weights.sum(0)>0).sum())}),
              flush=True)
    reports = []
    for held_index, held in enumerate(records):
        train = [e for i, e in enumerate(records) if i != held_index]
        for condition in ['both']:
            def features(record, selected_condition=condition):
                if selected_condition == 'both':
                    return np.concatenate([record['retinal'], record['motion']], axis=1)
                return record[selected_condition]
            x = np.concatenate([features(e) for e in train])
            y = np.concatenate([e['y'] for e in train])
            valid = np.concatenate([e['valid'] for e in train])
            mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
            d = np.clip((x-mean)/scale, -5, 5)[valid]
            centre = float(y[valid].mean())
            gram = d.T @ d
            gram.flat[::len(gram)+1] += len(d)
            right = d.T @ (y[valid]-centre)
            w = np.linalg.solve(gram, right)
            error = float(np.linalg.norm(gram @ w-right)/max(np.linalg.norm(right), 1e-12))
            if not np.isfinite(w).all() or error > 1e-8:
                raise ValueError('Ridge solve failed')
            truth = held['y'][held['valid']]
            original = features(held)
            split = held['retinal'].shape[1]
            interventions = [('intact', original)]
            for view, label in enumerate(['front', 'left', 'right', 'rear']):
                modified = original.copy()
                # Flattening order is view, neuron type, fixed image bin.
                retina_slice = slice(view*6*16, (view+1)*6*16)
                motion_slice = slice(split+view*8*16, split+(view+1)*8*16)
                modified[:, retina_slice] = mean[retina_slice]
                modified[:, motion_slice] = mean[motion_slice]
                interventions.append(('mean_'+label, modified))
            predictions = {}
            intact_pred = None
            for label, z in interventions:
                pred = centre + np.clip((z[held['valid']]-mean)/scale, -5, 5) @ w
                if label == 'intact':
                    intact_pred = pred.copy()
                groups = {}
                for group, mask in [('negative', truth < -.001), ('positive', truth > .001)]:
                    groups[group] = {'rows': int(mask.sum()), 'sign_accuracy': float(np.mean(
                        np.sign(pred[mask]) == np.sign(truth[mask]))) if mask.any() else None}
                predictions[label] = {'rmse_per_m': float(np.sqrt(np.mean((pred-truth)**2))),
                    'mean_abs_prediction_change_per_m': float(np.mean(np.abs(pred-intact_pred))),
                    'direction_groups': groups}
            reports.append({'held_out': held['id'], 'conditions': predictions,
                'constant_train_mean_rmse_per_m': float(np.sqrt(np.mean((centre-truth)**2))),
                'valid_rows': int(len(truth)), 'ridge_relative_solve_error': error})
            print(json.dumps(reports[-1]), flush=True)
    if any(sha256(path) != expected for path, expected in pins.items()):
        raise ValueError('Original inputs changed during experiment')
    write_json(run/'result.json', {'status': 'completed_four_view_interventions',
        'reports': reports, 'source_hashes': pins, 'development_opened': False,
        'image_grid': '4x4 bins of fixed recorded image_uv in each of four views',
        'feature_inputs': 'saved author retinal and/or motion activity only',
        'body_input_or_tilt_registration': False, 'original_author_optimizer_updates': 0,
        'intervention': 'one view retinal and motion features at training mean; others intact',
        'view_order': ['front', 'left', 'right', 'rear'],
        'diagnostic_direction_threshold_per_m': .001, 'rows_are_not_independent': True,
        'target': '20m teacher-path chord curvature, not optimal line or corner distance',
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
