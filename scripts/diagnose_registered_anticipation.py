"""Report every target/body-availability group; distinguish lookahead from current-turn sensing."""

import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_exterior_preview_learning import read
from scripts.train_visual_body_preview import direct_body_reference


def anticipation_groups(current, target, usable, valid):
    current, target = np.asarray(current), np.asarray(target)
    usable, valid = np.asarray(usable), np.asarray(valid)
    if (current.ndim != 1 or target.shape != current.shape or usable.shape != current.shape
            or valid.shape != current.shape or usable.dtype != bool or valid.dtype != bool
            or np.any(usable & ~valid) or not np.isfinite(current[usable]).all()
            or not np.isfinite(target[valid]).all()):
        raise ValueError('Matched finite valid diagnostic rows required')
    entering = usable & (np.abs(current) <= .001) & (np.abs(target) >= .003)
    exiting = usable & (np.abs(current) >= .003) & (np.abs(target) <= .001)
    opposite = usable & ~(entering | exiting) & (current*target < 0)
    opposite &= (np.abs(current) >= .001) & (np.abs(target) >= .001)
    groups = dict(entering=entering, exiting=exiting, opposite=opposite,
        other_valid=usable & ~(entering | exiting | opposite),
        unavailable_body=valid & ~usable, invalid_target=~valid)
    assert np.all(np.stack(list(groups.values())).sum(0) == 1)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--training', required=True)
    p.add_argument('--body-training', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    data, training, body_training = Path(a.data), Path(a.training), Path(a.body_training)
    source = read(data/'dataset.json')
    if (read(training/'manifest.json')['config']['dataset_sha256'] != sha256(data/'dataset.json')
            or read(training/'result.json')['status'] != (
                'completed_body_registered_preview_learning')):
        raise ValueError('Matching original completed registered experiment required')
    body_sources = read(body_training/'manifest.json')['config']['original_sources']
    if ({e['id']: (e['role'], e['targets_sha256']) for e in body_sources} != {
            e['id']: (e['role'], e['targets_sha256']) for e in source['episodes']}):
        raise ValueError('Same whole target histories and roles required for body comparison')
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        dataset_sha256=sha256(data/'dataset.json'), no_fitting=True,
        prediction_sha256=sha256(training/'registered-predictions.npz'),
        body_prediction_sha256=sha256(body_training/'body_only-predictions.npz'),
        evaluation_group_policy=('All disjoint groups plus full metrics reported; '
                                 'no filtering for fit or deployment'),
        straight_curvature_threshold_inv_m=.001, turn_curvature_threshold_inv_m=.003,
        group_definitions_are_posthoc_diagnostics=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports = []
    with load_array(training/'registered-predictions.npz', allow_pickle=False) as predictions, (
            load_array(body_training/'body_only-predictions.npz', allow_pickle=False)) as baseline:
        for e in source['episodes']:
            if e['role'] not in ('train', 'development'):
                raise ValueError('No evaluation access')
            if sha256(e['targets_file']) != e['targets_sha256']:
                raise ValueError('Original targets changed')
            with load_array(e['targets_file'], allow_pickle=False) as arrays:
                clock = arrays['observation_time']
                target, valid = arrays['preview'][:, 2, 1], arrays['preview_valid'][:, 2]
            body, mask, _ = direct_body_reference(e, clock,
                sha256('configs/ac-virtual-navigation-body-v2.json'))
            available = mask[:, 4] & mask[:, 16] & (body[:, 16] > 1)
            current = np.full(len(clock), np.nan)
            current[available] = -body[available, 4]/body[available, 16]
            usable = available & valid
            groups = anticipation_groups(current, target, usable, valid)
            inferred = dict(registered=predictions[e['id']+'__intact'][:, 5],
                neural_feature_mean=predictions[e['id']+'__neural_feature_mean'][:, 5],
                body_only=baseline[e['id']+'__intact'][:, 5], current_turn=current,
                zero=np.zeros(len(clock)))
            metrics = {}
            for name, selected in dict(full_valid=valid, fair_body_rows=usable, **groups).items():
                metrics[name] = dict(count=int(selected.sum()), rmse={})
                if name == 'invalid_target':
                    continue
                for model, values in inferred.items():
                    keep = selected & np.isfinite(values)
                    metrics[name]['rmse'][model] = dict(count=int(keep.sum()),
                        value=float(np.sqrt(np.mean((values[keep]-target[keep])**2)))
                        if keep.any() else None)
            reports.append(dict(id=e['id'], role=e['role'], rows=len(clock), metrics=metrics))
    write_json(run/'result.json', dict(status='completed_registered_anticipation_diagnosis',
        reports=reports, all_groups_partition_original_rows=True, optimizer_updates=0,
        original_auxiliary_gate_upgraded=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
