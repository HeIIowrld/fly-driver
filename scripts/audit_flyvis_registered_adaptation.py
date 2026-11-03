"""Audit original/free parameter boundaries and all final DEVELOPMENT predictions; no fit."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.turn_residual import current_turn_prior
from scripts.build_retinotopic_preview_dataset import read
from scripts.diagnose_registered_anticipation import anticipation_groups
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features, direct_body_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.training)
    result, manifest = read(root/'result.json'), read(root/'manifest.json')
    plan = manifest['config']['plan']
    if result['status'] != 'completed_author_registered_preview_adaptation':
        raise ValueError('Whole completed author adaptation required')
    for path, digest in plan['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Executed experiment input changed')
    source = read(Path(plan['data'])/'dataset.json')
    for entry in source['episodes']:
        if entry['role'] not in ('train', 'development'):
            raise ValueError('No evaluation access')
        for kind in ('features', 'targets'):
            if sha256(entry[kind+'_file']) != entry[kind+'_sha256']:
                raise ValueError('Whole original labels or reference features changed')
    train = [e for e in source['episodes'] if e['role'] == 'train']
    if [r['episode'] for r in result['records']] != [e['id'] for e in train]:
        raise ValueError('All declared original TRAIN histories required')
    expected_updates = 0
    for e, record in zip(train, result['records'], strict=True):
        with load_array(e['targets_file'], allow_pickle=False) as a:
            mask = np.repeat(a['preview_valid'], 2, axis=1)
            clock = a['observation_time']
        np.testing.assert_array_equal(record['gradient_target_counts'], mask[1:].sum(0))
        np.testing.assert_array_equal(record['detached_initial_target_counts'], mask[0].astype(int))
        if (record['observations'] != len(clock)
                or record['covered_original_intervals'] != len(clock)-1
                or record['recurrent_resets_inside_episode'] != 0):
            raise ValueError('Original observations/time intervals or state continuity changed')
        expected_updates += (len(clock)-1+plan['chunk_intervals']-1)//plan['chunk_intervals']
    if result['author_optimizer_updates'] != expected_updates:
        raise ValueError('Unexpected author update count')
    torch.set_num_threads(1)
    adapted = torch.load(root/'adapted-author.pt', map_location='cpu', weights_only=True)
    original = torch.load(Path(plan['model_dir'])/'best_chkpt', map_location='cpu',
                          weights_only=True)['network']
    flags = adapted['original_trainable_flags']
    if flags != read(root/'author-implementation.json')['original_trainable_flags']:
        raise ValueError('Original author trainability contract changed')
    changes, fixed_count = [], 0
    for key, old in original.items():
        new = adapted['network'][key]
        if not torch.isfinite(new).all():
            raise ValueError('Nonfinite adapted parameter')
        if not flags.get(key, False):
            if not torch.equal(old, new):
                raise ValueError('Originally fixed author value changed')
            fixed_count += old.numel()
        elif not torch.equal(old, new):
            changes.append(dict(name=key, elements=old.numel(),
                changed_elements=int(torch.count_nonzero(old-new)),
                max_absolute_change=float((old-new).abs().max())))
    changed_names = sorted(c['name'] for c in changes)
    if not changes or changed_names != sorted(result['author_parameters_changed']):
        raise ValueError('Recorded author adaptation not reproduced')
    saved = torch.load(plan['readout'], map_location='cpu', weights_only=False)
    readout = RegisteredPreview().eval().requires_grad_(False)
    readout.load_state_dict(saved['model'])
    dev = [e for e in source['episodes'] if e['role'] == 'development']
    if len(dev) != 1:
        raise ValueError('One known DEVELOPMENT history required')
    e = dev[0]
    with load_array(e['targets_file'], allow_pickle=False) as a:
        clock = a['observation_time']
        y, valid = a['preview'].reshape(-1, 6), np.repeat(a['preview_valid'], 2, axis=1)
    body, body_valid, _ = direct_body_reference(e, clock, sha256(plan['body_contract']))
    prior, available = current_turn_prior(body, body_valid)
    with load_array(e['features_file'], allow_pickle=False) as a:
        old_features = a['registered'].copy()
    with load_array(root/(e['id']+'-adapted-features.npz'), allow_pickle=False) as a:
        np.testing.assert_array_equal(a['observation_time'], clock)
        features = a['features'].copy()
    np.testing.assert_array_equal(features[:, 896:], old_features[:, 896:])
    b = body_features(body, body_valid, saved['body_mean'], saved['body_scale'],
                      saved['plan']['body_route_indices'])
    predictions, metrics = {}, {}
    with load_array(root/(e['id']+'-predictions.npz'), allow_pickle=False) as a:
        predictions = {k: a[k].copy() for k in a.files}
    np.testing.assert_array_equal(predictions['prior_only'], prior)
    feature_mean = ((features-saved['feature_mean'])/saved['feature_scale']).clip(-5, 5)
    feature_mean[:, :896] = 0
    raw_mean = features.copy()
    raw_mean[:, :896] = 0
    variants = dict(adapted=features, original=old_features, retinal_neural_mean=raw_mean)
    max_error = 0.
    for key, x in dict(**variants, neural_feature_mean=feature_mean).items():
        normalized = x if key == 'neural_feature_mean' else (
            (x-saved['feature_mean'])/saved['feature_scale']).clip(-5, 5)
        with torch.no_grad():
            inferred = readout(torch.tensor(normalized, dtype=torch.float32), torch.tensor(b))
        inferred = inferred.numpy()*saved['target_scale']+saved['target_mean']
        error = float(np.abs(inferred-predictions[key]).max())
        max_error = max(max_error, error)
        if error > 2e-5:
            raise ValueError('Independent CPU readout differs from GPU predictions')
    baseline_file = Path(plan['readout']).parent/'registered-predictions.npz'
    with load_array(baseline_file, allow_pickle=False) as a:
        original_error = float(np.abs(a[e['id']+'__intact']-predictions['original']).max())
    if original_error > 2e-6:
        raise ValueError('Original baseline differs from original matched experiment')
    for key, inferred in predictions.items():
        if inferred.shape != y.shape or not np.isfinite(inferred).all():
            raise ValueError('Whole finite prediction history required')
        errors = inferred-y
        rmse = [float(np.sqrt(np.mean(errors[valid[:, h], h]**2))) for h in range(6)]
        score = float(np.mean([np.mean((errors[valid[:, h], h]/saved['target_scale'][h])**2)
                               for h in range(6)]))
        recorded = next(r for r in result['reports'] if r['intervention'] == key)
        np.testing.assert_allclose(score, recorded['standardized_mse'], rtol=1e-12)
        np.testing.assert_allclose(rmse, [h['rmse'] for h in recorded['full']], rtol=1e-12)
        groups = anticipation_groups(prior[:, 5], y[:, 5], available & valid[:, 5], valid[:, 5])
        grouped = {}
        for name, keep in dict(full_valid=valid[:, 5], fair_body_rows=available & valid[:, 5],
                               **groups).items():
            grouped[name] = dict(count=int(keep.sum()),
                rmse=float(np.sqrt(np.mean(errors[keep, 5]**2)))
                if keep.any() and name != 'invalid_target' else None)
        metrics[key] = dict(standardized_mse=score, rmse=rmse, groups20=grouped)
    mse_gates = {key: metrics['adapted']['standardized_mse'] < .99*metrics[key]['standardized_mse']
                 for key in ('original', 'prior_only', 'neural_feature_mean',
                             'retinal_neural_mean')}
    bearings = []
    for h in (0, 2, 4):
        zero = np.sqrt(np.mean(y[valid[:, h], h]**2))
        mean = np.sqrt(np.mean((y[valid[:, h], h]-saved['target_mean'][h])**2))
        bearings.append(bool(metrics['adapted']['rmse'][h] < min(zero, mean,
            metrics['original']['rmse'][h], metrics['prior_only']['rmse'][h])))
    run = begin_run(args.output, dict(arguments=vars(args), no_fitting=True,
        program_sha256=sha256(__file__), training_result_sha256=sha256(root/'result.json'),
        learned_checkpoint_sha256=sha256(root/'adapted-author.pt'),
        prediction_sha256=sha256(root/(e['id']+'-predictions.npz')),
        feature_sha256=sha256(root/(e['id']+'-adapted-features.npz')),
        original_prediction_sha256=sha256(baseline_file)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'result.json', dict(status='completed_author_registered_adaptation_audit',
        author_optimizer_updates=expected_updates, changed_parameters=changes,
        fixed_parameter_elements_unchanged=fixed_count, independent_readout_error=max_error,
        original_baseline_error=original_error, observations=len(clock), metrics=metrics,
        mse_gates=mse_gates, bearing_gates=bearings,
        candidate_gate_passed=bool(all(mse_gates.values()) and all(bearings)),
        known_development_diagnostic_only=True, optimizer_updates=0, actor_deployed=False,
        motor_transfer_qualified=False, physiological_function_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
