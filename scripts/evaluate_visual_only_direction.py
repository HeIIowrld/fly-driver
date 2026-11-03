"""Whole-history TRAIN folds: can stored fly-vision features predict preview direction?"""

import argparse
import json

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_visual_residual_preview import open_episode, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Fixed source plan changed')
    plan = read(args.plan)
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Source changed: ' + path)
    entries = [e for e in read(plan['dataset'])['episodes'] if e['role'] == 'train']
    if [e['id'] for e in entries] != ['episode-04-train', 'episode-05-train',
                                      'episode-06-train']:
        raise ValueError('Original three TRAIN histories required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'plan_sha256': args.plan_sha256, 'development_opened': False,
        'direct_body_or_teacher_input_to_readout': False,
        'registered_features_depend_on_body_tilt': True, 'actor_deployed': False})
    episodes = [open_episode(e, plan['body_contract']) for e in entries]
    reports = []
    threshold = .001  # Diagnostic categorization, not a road-geometry safety threshold.
    for index, held in enumerate(episodes):
        train = [e for i, e in enumerate(episodes) if i != index]
        x = np.concatenate([e['x'][:, :896] for e in train]).astype(np.float64)
        valid = np.concatenate([e['valid'][:, 5] for e in train])
        y = np.concatenate([e['y'][:, 5] for e in train]).astype(np.float64)
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-5)
        design = np.clip((x-mean)/scale, -5, 5)[valid]
        centre = float(y[valid].mean())
        gram = design.T @ design
        gram.flat[::len(gram)+1] += len(design)
        right = design.T @ (y[valid]-centre)
        weights = np.linalg.solve(gram, right)
        solve_error = float(np.linalg.norm(gram @ weights-right)/max(np.linalg.norm(right), 1e-12))
        if not np.isfinite(weights).all() or solve_error > 1e-8:
            raise ValueError('Ridge solve failed')
        chosen = held['valid'][:, 5]
        truth = held['y'][chosen, 5]
        prediction = centre + np.clip((held['x'][chosen, :896]-mean)/scale, -5, 5) @ weights
        groups = {}
        for name, mask in [('negative', truth < -threshold),
                           ('near_zero', np.abs(truth) <= threshold),
                           ('positive', truth > threshold)]:
            groups[name] = {'rows': int(mask.sum()),
                'same_sign_fraction': (float(np.mean(np.sign(prediction[mask])
                                                     == np.sign(truth[mask])))
                                       if mask.any() else None),
                'mean_absolute_error_per_m': (float(np.mean(np.abs(prediction[mask]-truth[mask])))
                                              if mask.any() else None)}
        curved = np.abs(truth) > threshold
        reports.append({'held_out': held['id'], 'fit_histories': [e['id'] for e in train],
            'valid_rows': int(chosen.sum()), 'ridge_relative_solve_error': solve_error,
            'visual_only_rmse_per_m': float(np.sqrt(np.mean((prediction-truth)**2))),
            'constant_train_mean_rmse_per_m': float(np.sqrt(np.mean((centre-truth)**2))),
            'curved_rows_sign_accuracy': float(np.mean(
                np.sign(prediction[curved]) == np.sign(truth[curved]))) if curved.any() else None,
            'groups': groups})
        print(json.dumps(reports[-1]), flush=True)
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Source changed during fold experiment: ' + path)
    write_json(run/'result.json', {'status': 'completed_visual_only_train_history_folds',
        'reports': reports, 'features': 'first 896 saved retinal and motion features',
        'direct_body_features_at_readout': False, 'geometry_support_flags': False,
        'registered_features_depend_on_body_tilt': True,
        'constant_baseline': 'mean target curvature of the two fitted histories',
        'threshold_per_m_is_posthoc_diagnostic': threshold,
        'teacher_target': '20m teacher-path chord curvature; not certified optimal line',
        'development_opened': False, 'original_author_optimizer_updates': 0,
        'rows_are_not_independent': True, 'corner_distance_not_measured': True,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
