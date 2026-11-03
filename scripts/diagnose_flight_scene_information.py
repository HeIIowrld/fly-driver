"""Post-hoc feature-information controls, never a replacement driver or success gate."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def metrics(prediction, target, valid):
    return dict(count=int(valid.sum()),
        rmse=float(np.sqrt(np.mean((prediction[valid]-target[valid])**2))),
        prediction_mean=float(prediction[valid].mean()), target_mean=float(target[valid].mean()),
        outside_normalized_range=int((np.abs(prediction[valid]) > 1).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, training = Path(args.data), Path(args.training)
    receipt = read(training/'result.json')
    original_checkpoint = sha256(receipt['checkpoint'])
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
        dataset_sha256=sha256(root/'dataset.json'), checkpoint_sha256=original_checkpoint,
        diagnostic_regularizer=10., variance_floor=.005, feature_clip=5,
        policy='Train-only ridge controls; known development diagnosis, no tuning/deployment',
        script_sha256=sha256(__file__)))
    episodes = []
    for entry in read(root/'dataset.json')['episodes']:
        assert entry['role'] in ('train', 'development')
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        episodes.append((entry, arrays))
    raw_features, targets, startup = [], [], []
    for entry, a in episodes:
        body = np.concatenate([np.where(a['body_valid'], a['body'], 0), a['body_valid']], axis=1)
        x = np.concatenate([a['scene_features'], body], axis=1).astype(np.float64)
        raw_features.append(x)
        if entry['role'] != 'train':
            continue
        valid = a['target_valid'][:, 0]
        targets.append(a['target'][valid, 0].astype(np.float64))
        elapsed = a['observation_time']-a['observation_time'][0]
        startup.append(dict(file=entry['file'], phases=[dict(
            phase=name, left=int((selected & valid & (a['target'][:, 0] > .01)).sum()),
            right=int((selected & valid & (a['target'][:, 0] < -.01)).sum()),
            neutral=int((selected & valid & (np.abs(a['target'][:, 0]) <= .01)).sum()))
            for name, selected in [('first3s', elapsed < 3), ('after3s', elapsed >= 3)]]))
    train_x = np.concatenate([x[a['target_valid'][:, 0]]
        for x, (e, a) in zip(raw_features, episodes, strict=True) if e['role'] == 'train'])
    train_y = np.concatenate(targets)
    mean, scale = train_x.mean(0), np.maximum(train_x.std(0), .005)
    normalized = [np.clip((x-mean)/scale, -5, 5) for x in raw_features]
    train_x = np.clip((train_x-mean)/scale, -5, 5)
    masks = {}
    for name in ('scene', 'scene_balanced', 'intensity', 'motion', 'body', 'scene_body'):
        mask = np.zeros(train_x.shape[1], dtype=bool)
        if name.startswith('scene'):
            mask[:576] = True
        if name in ('intensity', 'motion'):
            channels = mask[:576].reshape(4, 9, 16)
            if name == 'intensity':
                channels[:, :1] = True
            else:
                channels[:, 1:] = True
        if name in ('body', 'scene_body'):
            mask[576:] = True
        masks[name] = mask
    reports, predictions, coefficients = {}, {}, {}
    for name, mask in masks.items():
        x = np.column_stack([np.ones(len(train_x)), train_x[:, mask]])
        weight = np.ones(len(train_y), dtype=np.float64)
        if name == 'scene_balanced':
            groups = np.where(train_y > .01, 0, np.where(train_y < -.01, 1, 2))
            counts = np.bincount(groups, minlength=3)
            assert (counts > 0).all()
            weight = len(train_y)/(3*counts[groups])
        penalty = np.eye(x.shape[1])*10.
        penalty[0, 0] = 0
        coefficient = np.linalg.solve(x.T@(weight[:, None]*x)+penalty, x.T@(weight*train_y))
        coefficients[name] = coefficient
        reports[name] = []
        for index, (entry, a) in enumerate(episodes):
            estimate = np.column_stack([np.ones(len(a['target'])),
                                       normalized[index][:, mask]])@coefficient
            valid = a['target_valid'][:, 0]
            late = valid & (a['observation_time']-a['observation_time'][0] >= 3)
            reports[name].append(dict(file=entry['file'], role=entry['role'],
                full=metrics(estimate, a['target'][:, 0], valid),
                after3s=metrics(estimate, a['target'][:, 0], late)))
            predictions[name+'_'+entry['file'].removesuffix('.npz')] = estimate
    np.savez_compressed(run/'diagnostic-predictions.npz', **predictions)
    np.savez_compressed(run/'diagnostic-coefficients.npz', **coefficients, mean=mean, scale=scale)
    assert sha256(receipt['checkpoint']) == original_checkpoint
    write_json(run/'result.json', dict(status='completed_posthoc_scene_information_diagnosis',
        metrics=reports, training_startup_strata=startup, ridge_regularizer=10.,
        learned_diagnostic_controls_not_fly_circuit=True, diagnostic_only_no_deployment=True,
        checkpoint_modified=False, evaluation_role_arrays_opened=False,
        source_arrays_or_times_modified=False, neural_optimizer_updates=0,
        ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
