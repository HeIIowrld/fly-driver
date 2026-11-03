"""Original retinal versus coarse spatial information; diagnostic controls never drive."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def weights(uv, basis):
    x, y = np.meshgrid((np.arange(basis['columns'])+.5)/basis['columns'],
                       (np.arange(basis['rows'])+.5)/basis['rows'])
    centers = np.column_stack([x.ravel(), y.ravel()])
    value = np.exp(-np.sum((centers[:, None]-uv[None])**2, axis=-1)
                   /(2*basis['sigma_image_uv']**2))
    return (value/value.sum(axis=1, keepdims=True)).astype(np.float32)


def metrics(prediction, target, valid):
    return dict(count=int(valid.sum()),
        rmse=float(np.sqrt(np.mean((prediction[valid]-target[valid])**2))),
        prediction_mean=float(prediction[valid].mean()), target_mean=float(target[valid].mean()),
        outside_normalized_range=int((np.abs(prediction[valid]) > 1).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan['data'])
    training = read(Path(plan['training'])/'result.json')
    checkpoint_hash = sha256(training['checkpoint'])
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
        plan_sha256=sha256(args.plan), script_sha256=sha256(__file__),
        dataset_sha256=sha256(root/'dataset.json'), checkpoint_sha256=checkpoint_hash,
        posthoc_diagnosis_not_success_gate=True, optimizer_updates=0,
        ui_actions_sent=0, actuator_messages_sent=0))
    episodes, proof = [], []
    reference_uv, pool4, pool8, row_basis = None, None, None, None
    for entry in read(root/'dataset.json')['episodes']:
        assert entry['role'] in ('train', 'development')
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        visual = Path(entry['original']['scene_replay'])
        receipt = read(visual/'result.json')
        assert sha256(visual/'responses.npz') == entry['source_visual_sha256']
        assert receipt['responses_sha256'] == entry['source_visual_sha256']
        with load_array(visual/'retina-registration.npz', allow_pickle=False) as archive:
            uv = archive['image_uv'].copy()
        if reference_uv is None:
            reference_uv = uv
            pool4, pool8 = weights(uv, plan['pool4']), weights(uv, plan['pool8'])
            _, singular, vectors = np.linalg.svd(pool4.astype(np.float64), full_matrices=False)
            row_basis = vectors[singular > singular[0]*1e-10]
        else:
            np.testing.assert_array_equal(reference_uv, uv)
        with load_array(visual/'responses.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(a['observation_time'],
                                          archive['decision_monotonic_seconds'])
            scene = np.concatenate([archive['retina_intensity'][:, :, None],
                                    archive['motion_activity']], axis=2)
        bounded = np.tanh(scene)
        coarse = bounded@pool4.T
        error = float(np.abs(coarse.reshape(len(scene), -1)-a['scene_features']).max())
        assert error <= plan['cache_reproduction_tolerance']
        channel_proof = []
        for channel in range(9):
            # Mathematical information-loss diagnostic, not an encoding fitted on labels.
            x = bounded[:, :, channel].astype(np.float64)
            centered = x-x.mean(axis=0, keepdims=True)
            energy = float(np.square(centered).sum())
            represented = float(np.square(centered@row_basis.T).sum())
            channel_proof.append(dict(channel=channel,
                abs_raw_above3_fraction=float((np.abs(scene[:, :, channel]) > 3).mean()),
                raw_temporal_std=float(scene[:, :, channel].std(axis=0).mean()),
                tanh_temporal_std=float(bounded[:, :, channel].std(axis=0).mean()),
                pool4_rowspace_temporal_energy_fraction=represented/energy if energy else None))
        road = ((uv[:, 1] >= plan['roadband']['v_min'])
                & (uv[:, 1] <= plan['roadband']['v_max']))
        front = bounded[:, 0]
        fine = front@pool8.T
        features = dict(existing_all_scene_576=a['scene_features'],
            front_intensity_pool4=coarse[:, 0, 0], front_intensity_pool8=fine[:, 0],
            front_intensity_raw=front[:, 0], front_roadband_intensity_raw=front[:, 0, road],
            front_motion_pool4=coarse[:, 0, 1:].reshape(len(scene), -1),
            front_motion_pool8=fine[:, 1:].reshape(len(scene), -1),
            front_joint_pool8=fine.reshape(len(scene), -1))
        episodes.append((entry, a, features))
        proof.append(dict(file=entry['file'], role=entry['role'], observations=len(scene),
            raw_visual_sha256=entry['source_visual_sha256'], cache_reproduction_max_error=error,
            channels=channel_proof))
        write_json(run/'progress.json', dict(stage='source_alignment', complete=len(episodes)))
    reports, predictions, fitted = {}, {}, {}
    for variant in plan['variants']:
        train_frames = np.concatenate([f[variant] for e, _, f in episodes if e['role'] == 'train'])
        mean = train_frames.mean(0, dtype=np.float64)
        scale = np.maximum(train_frames.astype(np.float64).std(0), plan['normalization_floor'])
        matrices, labels = [], []
        for e, a, f in episodes:
            if e['role'] != 'train':
                continue
            valid = a['target_valid'][:, 0]
            matrices.append(np.clip((f[variant][valid]-mean)/scale, -plan['feature_clip'],
                                    plan['feature_clip']))
            labels.append(a['target'][valid, 0].astype(np.float64))
        x = np.concatenate(matrices)
        y = np.concatenate(labels)
        x = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(x.shape[1])*plan['ridge_penalty']
        penalty[0, 0] = 0
        coefficient = np.linalg.solve(x.T@x+penalty, x.T@y)
        reports[variant] = []
        fitted[variant+'_coefficient'] = coefficient
        fitted[variant+'_mean'], fitted[variant+'_scale'] = mean, scale
        for e, a, f in episodes:
            normalized = np.clip((f[variant]-mean)/scale,
                                 -plan['feature_clip'], plan['feature_clip'])
            estimate = np.column_stack([np.ones(len(normalized)), normalized])@coefficient
            valid = a['target_valid'][:, 0]
            late = valid & (a['observation_time']-a['observation_time'][0] >= 3)
            reports[variant].append(dict(file=e['file'], role=e['role'], features=len(mean),
                full=metrics(estimate, a['target'][:, 0], valid),
                after3s=metrics(estimate, a['target'][:, 0], late)))
            predictions[variant+'_'+e['file'].removesuffix('.npz')] = estimate
        write_json(run/'progress.json', dict(stage='diagnostic_controls', complete=list(reports)))
    np.savez_compressed(run/'diagnostic-predictions.npz', **predictions)
    np.savez_compressed(run/'diagnostic-coefficients.npz', **fitted)
    assert sha256(training['checkpoint']) == checkpoint_hash
    write_json(run/'result.json', dict(status='completed_spatial_information_diagnosis',
        source_alignment_and_spatial_energy=proof, metrics=reports,
        pooling_rank=int(len(row_basis)), original_retinal_positions=721,
        coarse_cache_max_error=max(p['cache_reproduction_max_error'] for p in proof),
        evaluation_role_arrays_opened=False, labels_or_clocks_modified=False,
        fitted_controls_not_fly_circuit=True, neural_checkpoint_modified=False,
        diagnostic_only_no_driver_qualification=True, goal_achieved=False))


if __name__ == '__main__':
    main()
