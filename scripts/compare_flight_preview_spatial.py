"""Compare spatially preserved retinal responses on unchanged future-path targets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    data, preview = Path(plan['data']), Path(plan['preview'])
    dataset, labels = read(data/'dataset.json'), read(preview/'result.json')
    labelled = {e['file']: e for e in labels['episodes']}
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), dataset_sha256=sha256(data/'dataset.json'),
        preview_result_sha256=sha256(preview/'result.json'),
        pooling_module_sha256=sha256('src/flybrain_drive/flight_circuit.py'),
        ui_actions_sent=0, actuator_messages_sent=0, actor_training=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'executed-pooling-module.py').write_bytes(
        Path('src/flybrain_drive/flight_circuit.py').read_bytes())
    (run/'plan.json').write_bytes(Path(args.plan).read_bytes())
    episodes, proofs = [], []
    reference_uv = None
    assert torch.cuda.is_available(), 'Original GPU pooling path required'
    for entry in dataset['episodes']:
        assert entry['role'] in ('train', 'development')
        label = labelled[entry['file']]
        assert label['role'] == entry['role']
        assert sha256(data/entry['file']) == entry['sha256']
        assert sha256(preview/entry['file']) == label['auxiliary_sha256']
        with load_array(data/entry['file'], allow_pickle=False) as a:
            scene, clock = a['scene_features'].copy(), a['observation_time'].copy()
        with load_array(preview/entry['file'], allow_pickle=False) as a:
            y, valid = a['preview'].copy(), a['preview_valid'].copy()
            np.testing.assert_array_equal(clock, a['observation_time'])
        visual = Path(entry['original']['scene_replay'])
        assert sha256(visual/'responses.npz') == entry['source_visual_sha256']
        receipt = read(visual/'result.json')
        assert receipt['responses_sha256'] == entry['source_visual_sha256']
        with load_array(visual/'retina-registration.npz', allow_pickle=False) as a:
            uv = a['image_uv'].copy()
        if reference_uv is None:
            reference_uv = uv
            pool4 = FlightScenePool(uv, plan['pool4']).to('cuda')
            pool8 = FlightScenePool(uv, plan['pool8']).to('cuda')
        else:
            np.testing.assert_array_equal(uv, reference_uv)
        with load_array(visual/'responses.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['decision_monotonic_seconds'])
            raw = np.concatenate([a['retina_intensity'][:, :, None],
                                  a['motion_activity']], axis=2)
        with torch.no_grad():
            tensor = torch.tensor(raw, dtype=torch.float32, device='cuda')
            coarse = pool4(tensor).cpu().numpy().reshape(len(scene), 4, 9, 16)
            fine = pool8(tensor).cpu().numpy().reshape(len(scene), 4, 9, 64)[:, 0]
            bounded = torch.tanh(tensor[:, 0, 0]).cpu().numpy()
        error = float(np.abs(coarse.reshape(len(scene), -1)-scene).max())
        assert error <= plan['cache_reproduction_tolerance']
        feature = dict(existing_all_scene_576=scene,
            front_joint_pool4=coarse[:, 0].reshape(len(scene), -1),
            front_intensity_pool8=fine[:, 0], front_intensity_raw=bounded,
            front_motion_pool8=fine[:, 1:].reshape(len(scene), -1),
            front_joint_pool8=fine.reshape(len(scene), -1))
        episodes.append(dict(file=entry['file'], role=entry['role'], feature=feature,
                             y=y, valid=valid, clock=clock))
        proofs.append(dict(file=entry['file'], role=entry['role'], observations=len(scene),
            visual_sha256=entry['source_visual_sha256'], cache_max_error=error,
            retina_registration_sha256=sha256(visual/'retina-registration.npz'),
            parent_episode_sha256=entry['sha256'], auxiliary_sha256=label['auxiliary_sha256']))
        write_json(run/'progress.json', dict(stage='features', completed=len(episodes)))
    train = [e for e in episodes if e['role'] == 'train']
    ty, tm = (np.concatenate([e[k] for e in train]) for k in ('y', 'valid'))
    fitted, predictions, metrics = {}, {}, []
    for variant in plan['variants']:
        x = np.concatenate([e['feature'][variant] for e in train]).astype(float)
        mean, scale = x.mean(0), x.std(0).clip(plan['normalization_floor'])
        z = ((x-mean)/scale).clip(-plan['feature_clip'], plan['feature_clip'])
        coefficients, intercepts, target_means = [], [], []
        for h in range(3):
            xx, yy = z[tm[:, h]], ty[tm[:, h], h]
            mx, my = xx.mean(0), yy.mean(0)
            centered = xx-mx
            coefficient = np.linalg.solve(centered.T@centered/len(xx)
                + np.eye(x.shape[1])*plan['ridge_mean_loss_penalty'],
                centered.T@(yy-my)/len(xx))
            coefficients.append(coefficient)
            intercepts.append(my-mx@coefficient)
            target_means.append(my)
        coefficients, intercepts = np.array(coefficients), np.array(intercepts)
        fitted.update({variant+'_mean': mean, variant+'_scale': scale,
                       variant+'_weights': coefficients, variant+'_bias': intercepts})
        for e in episodes:
            z = ((e['feature'][variant]-mean)/scale).clip(
                -plan['feature_clip'], plan['feature_clip'])
            estimate = np.einsum('nf,hfc->nhc', z, coefficients)+intercepts
            predictions[variant+'_'+e['file'].removesuffix('.npz')] = estimate
            rows = []
            for h in range(3):
                row = dict(distance_m=[5, 10, 20][h])
                for name, keep in [('full', e['valid'][:, h]),
                        ('after3s', e['valid'][:, h] & (e['clock'] >= e['clock'][0]+3))]:
                    y, pred = e['y'][keep, h], estimate[keep, h]
                    row[name] = dict(n=int(keep.sum()),
                        rmse=np.sqrt(((pred-y)**2).mean(0)).tolist(),
                        train_mean_rmse=np.sqrt(((target_means[h]-y)**2).mean(0)).tolist(),
                        zero_rmse=np.sqrt((y**2).mean(0)).tolist())
                rows.append(row)
            metrics.append(dict(variant=variant, features=len(mean), file=e['file'],
                                role=e['role'], metrics=rows))
        write_json(run/'progress.json', dict(stage='probe', completed_variant=variant))
    np.savez_compressed(run/'coefficients.npz', **fitted)
    np.savez_compressed(run/'predictions.npz', **predictions)
    write_json(run/'result.json', dict(status='completed_spatial_preview_comparison',
        source_proof=proofs, metrics=metrics, train_only_normalization_and_fit=True,
        original_roles_labels_and_clocks_unchanged=True, evaluation_role_arrays_opened=False,
        physiological_claim=False, driving_actor_qualified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
