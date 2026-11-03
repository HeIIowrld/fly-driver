"""Apply the preregistered auxiliary gate; this never qualifies motor control or deployment."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def auxiliary_gate(intact, ablated, scale):
    scale = np.asarray(scale, float)
    if (len(intact) != 6 or len(ablated) != 6 or scale.shape != (6,)
            or not np.isfinite(scale).all() or not (scale > 0).all()
            or any(i['count'] <= 0 or i['count'] != a['count']
                   for i, a in zip(intact, ablated, strict=True))):
        raise ValueError('Six whole valid matched heads and TRAIN-only scales required')
    numbers = [[r[k] for k in ('rmse', 'train_mean_rmse', 'zero_rmse')]
               for r in intact+ablated]
    if not np.isfinite(numbers).all() or np.any(np.asarray(numbers) < 0):
        raise ValueError('Finite nonnegative errors required')
    bearing = [intact[h]['rmse'] < min(intact[h]['train_mean_rmse'], intact[h]['zero_rmse'])
               for h in (0, 2, 4)]
    actual = float(np.mean([(r['rmse']/s)**2 for r, s in zip(intact, scale, strict=True)]))
    removed = float(np.mean([(r['rmse']/s)**2 for r, s in zip(ablated, scale, strict=True)]))
    return dict(bearing_beats_both_baselines=bearing, intact_standardized_mse=actual,
        vision_removed_standardized_mse=removed, vision_contribution=actual < removed,
        auxiliary_gate_passed=all(bearing) and actual < removed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--training', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan, data, training = read(a.plan), Path(a.data), Path(a.training)
    dataset, manifest, result = read(data/'dataset.json'), read(training/'manifest.json'), read(
        training/'result.json')
    provenance = manifest['config']
    if (provenance['plan_sha256'] != sha256(a.plan)
            or provenance['dataset_sha256'] != sha256(data/'dataset.json')
            or provenance['program_sha256'] != plan['source_sha256'][
                'scripts/train_spatial_preview_boundary.py']
            or plan['preregistered_candidate'] != 'processed_actual'
            or result['status'] != 'completed_paired_spatial_preview_learning'
            or any(r['epochs'] != 200 for r in result['records'])
            or sorted(e['role'] for e in dataset['episodes']) != [
                'development', 'train', 'train', 'train']):
        raise ValueError('Declared whole new-mount paired learning and original roles required')
    train_y, train_valid = [], []
    for entry in dataset['episodes']:
        if sha256(entry['targets_file']) != entry['targets_sha256']:
            raise ValueError('Original targets changed')
        if entry['role'] == 'train':
            with load_array(entry['targets_file'], allow_pickle=False) as array:
                train_y.append(array['preview'].reshape(-1, 6))
                train_valid.append(np.repeat(array['preview_valid'], 2, axis=1))
    y, valid = np.concatenate(train_y), np.concatenate(train_valid)
    scales = np.maximum([y[valid[:, h], h].std() for h in range(6)], plan['target_scale_floor'])
    development = next(e['id'] for e in dataset['episodes'] if e['role'] == 'development')
    reports = {r['intervention']: r for r in result['reports'] if r['condition'] == (
        'processed_actual') and r['episode'] == development and r['role'] == 'development'}
    gate = auxiliary_gate(reports['intact']['full'], reports['vision_train_mean']['full'], scales)
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        training_result_sha256=sha256(training/'result.json'), plan_sha256=sha256(a.plan),
        dataset_sha256=sha256(data/'dataset.json'), no_fitting=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'result.json', dict(status='completed_auxiliary_candidate_audit', **gate,
        candidate='processed_actual', development=development,
        train_only_target_scale=scales.tolist(), development_used_to_fit=False,
        physiological_function_verified=False, motor_transfer_qualified=False,
        actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
