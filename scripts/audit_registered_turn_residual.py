"""Independently recompute whole-history residual diagnostic gates; never fit."""

import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.turn_residual import current_turn_prior
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_registered_turn_residual import CONDITIONS
from scripts.train_visual_body_preview import direct_body_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training = Path(args.training)
    manifest, result = read(training/'manifest.json'), read(training/'result.json')
    plan = manifest['config']['plan']
    if (result['status'] != 'completed_registered_turn_residual'
            or result['optimizer_updates'] != 800 or plan['conditions'] != CONDITIONS
            or any(r['epochs'] != 200 for r in result['records'])
            or len({r['initial_parameter_sha256'] for r in result['records']}) != 1):
        raise ValueError('Complete fixed-last matched four-condition experiment required')
    source = Path(plan['data'])/'dataset.json'
    if sha256(source) != manifest['config']['dataset_sha256']:
        raise ValueError('Original dataset changed')
    episodes = read(source)['episodes']
    ys, ym = np.asarray(result['train_target_scale']), np.asarray(result['train_target_mean'])
    recomputed, originals, development = [], {}, {}
    for e in episodes:
        if e['role'] not in ('train', 'development'):
            raise ValueError('No evaluation access')
        if sha256(e['targets_file']) != e['targets_sha256']:
            raise ValueError('Original target changed')
        with load_array(e['targets_file'], allow_pickle=False) as a:
            y = a['preview'].reshape(-1, 6)
            valid = np.repeat(a['preview_valid'], 2, axis=1)
            clock = a['observation_time']
        body, mask, _ = direct_body_reference(e, clock, sha256(plan['body_contract']))
        prior, available = current_turn_prior(body, mask)
        originals[e['id']] = (y, valid, prior)
        for condition in CONDITIONS:
            with load_array(training/(condition+'-predictions.npz'), allow_pickle=False) as pred:
                keys = [key for key in pred.files if key.startswith(e['id']+'__')]
                for key in keys:
                    inferred = pred[key]
                    intervention = key.split('__')[1]
                    if inferred.shape != y.shape or not np.isfinite(inferred).all():
                        raise ValueError('Every original row and finite six-head output required')
                    if intervention == 'prior_only':
                        np.testing.assert_array_equal(inferred, prior)
                    errors = inferred-y
                    rmse = [float(np.sqrt(np.mean(errors[valid[:, h], h]**2)))
                            for h in range(6)]
                    score = float(np.mean([np.mean((errors[valid[:, h], h]/ys[h])**2)
                                           for h in range(6)]))
                    saved = next(r for r in result['reports'] if r['condition'] == condition
                        and r['episode'] == e['id'] and r['intervention'] == intervention)
                    np.testing.assert_allclose(score, saved['standardized_mse'], rtol=1e-12)
                    np.testing.assert_allclose(rmse, [h['rmse'] for h in saved['full']], rtol=1e-12)
                    record = dict(condition=condition, episode=e['id'], role=e['role'],
                        intervention=intervention, rows=len(y), valid_counts=valid.sum(0).tolist(),
                        prior_available_rows=int(available.sum()), standardized_mse=score,
                        rmse=rmse)
                    recomputed.append(record)
                    if e['role'] == 'development':
                        development[condition+'/'+intervention] = record
    dev_entries = [e for e in episodes if e['role'] == 'development']
    if len(dev_entries) != 1:
        raise ValueError('Exactly one declared known DEVELOPMENT history required')
    y, valid, _ = originals[dev_entries[0]['id']]
    candidate = development['registered_balanced/intact']
    comparisons = ['registered_balanced/prior_only', 'body_balanced/intact',
                   'registered_balanced/neural_feature_mean',
                   'registered_balanced/retinal_neural_mean']
    mse_gates = {key: candidate['standardized_mse'] < .99*development[key]['standardized_mse']
                 for key in comparisons}
    bearing_gates = []
    for h in (0, 2, 4):
        zero = np.sqrt(np.mean(y[valid[:, h], h]**2))
        mean = np.sqrt(np.mean((y[valid[:, h], h]-ym[h])**2))
        bearing_gates.append(bool(candidate['rmse'][h] < min(zero, mean,
            development['registered_balanced/prior_only']['rmse'][h],
            development['body_balanced/intact']['rmse'][h])))
    run = begin_run(args.output, dict(arguments=vars(args), no_fitting=True,
        program_sha256=sha256(__file__), training_result_sha256=sha256(training/'result.json'),
        original_sources=episodes, prediction_sha256={c: sha256(training/(c+'-predictions.npz'))
                                                      for c in CONDITIONS}))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'result.json', dict(status='completed_registered_turn_residual_audit',
        recomputed=recomputed, mse_gates=mse_gates, bearing_gates=bearing_gates,
        candidate_gate_passed=bool(all(mse_gates.values()) and all(bearing_gates)),
        known_development_diagnostic_only=True, optimizer_updates=0,
        motor_transfer_qualified=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
