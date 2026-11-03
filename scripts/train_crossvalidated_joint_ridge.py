"""Select one linear four-view readout by original TRAIN capture-group folds."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_joint_view_expanded_histories import prepare, read, score


def group(record):
    if record['id'].startswith('original-'):
        return 'original'
    if record['id'].startswith('spatial-'):
        return 'spatial'
    return 'exterior'


def fit(records, penalty):
    x = np.concatenate([r['x'] for r in records])
    y = np.concatenate([r['y'] for r in records])
    valid = np.concatenate([r['valid'] for r in records])
    if not valid.any():
        raise ValueError('Training target support required')
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    design = np.clip((x-mean)/scale, -5, 5)[valid]
    target_mean = float(y[valid].mean())
    gram = design.T @ design
    gram.flat[::len(gram)+1] += len(design)*penalty
    right = design.T @ (y[valid]-target_mean)
    weights = np.linalg.solve(gram, right)
    error = float(np.linalg.norm(gram @ weights-right)/max(np.linalg.norm(right), 1e-12))
    if not np.isfinite(weights).all() or error > 1e-8:
        raise ValueError('Finite accurate ridge solve required')
    return {'mean': mean, 'scale': scale, 'weights': weights, 'target_mean': target_mean,
            'solve_error': error}


def predict(model, record):
    return model['target_mean'] + np.clip(
        (record['x']-model['mean'])/model['scale'], -5, 5) @ model['weights']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--older-dataset', required=True)
    parser.add_argument('--older-readout', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Original source plan changed')
    plan = read(args.plan)
    entries = read(args.older_dataset)['episodes'] + read(plan['dataset'])['episodes']
    source_list = read(args.older_readout)['episodes'] + read(plan['sensor_replay'])['episodes']
    sources = {e['id']: e for e in source_list}
    if (len(sources) != len(entries) or len({e['id'] for e in entries}) != len(entries)
            or len({str(Path(sources[e['id']]['recording']).resolve()) for e in entries})
            != len(entries) or any(e['role'] != sources[e['id']]['role'] for e in entries)):
        raise ValueError('Unique original recordings and unchanged roles required')
    train = [e for e in entries if e['role'] == 'train']
    development = [e for e in entries if e['role'] == 'development']
    if len(train) != 11 or len(development) != 3:
        raise ValueError('Expected original 11 TRAIN and 3 known DEVELOPMENT histories')
    penalties = [.1, 1., 10.]
    held_groups = ['original', 'spatial', 'exterior']
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_plan_sha256': args.plan_sha256, 'older_dataset_sha256': sha256(args.older_dataset),
        'older_readout_sha256': sha256(args.older_readout), 'penalties': penalties,
        'selection': 'mean held-out TRAIN-history RMSE over three capture groups',
        'development_read_before_checkpoint': False, 'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    train_data = []
    for e in train:
        train_data.append(prepare(e, sources[e['id']]))
        print(json.dumps({'loaded_train': e['id']}), flush=True)
    for e in train_data[1:]:
        np.testing.assert_array_equal(train_data[0]['uv'], e['uv'])
    folds = []
    for held_group in held_groups:
        fitting = [e for e in train_data if group(e) != held_group]
        held = [e for e in train_data if group(e) == held_group]
        if not fitting or not held:
            raise ValueError('Every declared capture group must be represented')
        for penalty in penalties:
            model = fit(fitting, penalty)
            metrics = [dict(id=e['id'], **score(e, predict(model, e), model['target_mean']))
                       for e in held]
            folds.append({'held_group': held_group, 'penalty': penalty,
                          'fit_ids': [e['id'] for e in fitting], 'reports': metrics})
        print(json.dumps({'completed_fold': held_group}), flush=True)
    losses = {str(p): float(np.mean([r['rmse_per_m'] for f in folds if f['penalty'] == p
                                    for r in f['reports']])) for p in penalties}
    selected = min(penalties, key=lambda p: (losses[str(p)], p))
    model = fit(train_data, selected)
    checkpoint = run/'joint-view-ridge.npz'
    np.savez_compressed(checkpoint, feature_mean=model['mean'], feature_scale=model['scale'],
        weights=model['weights'], target_mean=model['target_mean'], penalty=selected)
    print(json.dumps({'stage': 'checkpoint_fixed', 'penalty': selected,
                      'train_fold_rmse': losses}), flush=True)
    dev_data = [prepare(e, sources[e['id']], train_data[0]['uv']) for e in development]
    reports = [dict(id=e['id'], role=e['role'], **score(e, predict(model, e), model['target_mean']))
               for e in dev_data]
    for e in [*train_data, *dev_data]:
        if any(sha256(path) != digest for path, digest in e['pins'].items()):
            raise ValueError('Source changed during experiment')
    result = {'status': 'completed_train_group_crossvalidated_joint_ridge',
        'folds': folds, 'candidate_penalties': penalties, 'train_fold_mean_rmse': losses,
        'selected_penalty': selected, 'development_reports': reports,
        'checkpoint_sha256': sha256(checkpoint), 'development_read_before_checkpoint': False,
        'known_development_not_independent_final_test': True, 'author_optimizer_updates': 0,
        'target': '20m teacher path chord curvature, not optimal line or corner distance',
        'four_view_retinal_and_motion_features': True, 'direct_body_or_map_input': False,
        'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps({'selected': selected, 'development_reports': reports}), flush=True)


if __name__ == '__main__':
    main()
