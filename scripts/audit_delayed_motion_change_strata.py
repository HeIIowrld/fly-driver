"""Stratify fixed delayed-motion replay by sensory change, never by teacher outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'delay', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    training, delay = Path(a.training), Path(a.delay)
    tr, tm = read(training/'result.json'), read(training/'manifest.json')
    dr, dm = read(delay/'result.json'), read(delay/'manifest.json')['config']
    root = Path(tm['config']['plan']['data'])
    data = read(root/'dataset.json')
    if (tr['status'] != 'completed_fixed_joint_command_training'
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or dr['status'] != 'completed_four_view_motion_unclamped_delay_replay'
            or sha256(delay/'predictions.npz') != dr['predictions_sha256']
            or dm['training_result_sha256'] != sha256(training/'result.json')
            or dm['dataset_sha256'] != sha256(root/'dataset.json')):
        raise ValueError('Exact fixed checkpoint and causal delay evidence required')
    checkpoint = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model']
    mean = checkpoint['backbone.scene_command.mean'].numpy().reshape(4, 9, 16)[:, 1:, :].ravel()
    scale = checkpoint['backbone.scene_command.scale'].numpy().reshape(4, 9, 16)[:, 1:, :].ravel()
    if (mean.shape != (512,) or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError('Exact TRAIN normalization required')
    entries = [e for e in data['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('Original known development histories required')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'checkpoint_sha256': tr['checkpoint_sha256'], 'delay_sha256': dr['predictions_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'),
        'strata': 'within-episode quartiles of RMS clipped normalized visual motion change',
        'stratum_selected_without_teacher': True, 'known_development_not_blind': True,
        'actor_deployed': False, 'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, saved = [], {}
    with load_array(delay/'predictions.npz', allow_pickle=False) as predictions:
        for entry in entries:
            source = root/entry['file']
            if sha256(source) != entry['sha256']:
                raise ValueError('Original role-bound episode changed')
            key = entry['file'].removesuffix('.npz')
            time, source_index = predictions[key+'_time'], predictions[key+'_source_2_0']
            eligible = predictions[key+'_score_mask'].copy()
            with load_array(source, allow_pickle=False) as episode:
                np.testing.assert_array_equal(time, episode['observation_time'])
                scene = episode['scene_features'].copy().reshape(-1, 4, 9, 16)[:, :, 1:, :]
            scene = scene.reshape(-1, 512)
            if (not np.isfinite(scene).all() or not eligible.any()
                    or np.any(source_index[eligible] < 0)
                    or np.any(source_index[eligible] >= np.arange(len(time))[eligible])):
                raise ValueError('Finite motion and strictly earlier sources required')
            standardized = np.clip((scene - mean) / scale, -5, 5)
            delta = np.zeros(len(time), dtype=np.float64)
            delta[eligible] = np.sqrt(np.mean((standardized[eligible]
                - standardized[source_index[eligible]])**2, axis=1))
            threshold = float(np.quantile(delta[eligible], .75))
            high, low = eligible & (delta > threshold), eligible & (delta <= threshold)
            if not high.any() or not low.any():
                raise ValueError('Distinct nonempty independently selected change strata required')
            # First open labels only after sensory stratum selection is fixed.
            with load_array(source, allow_pickle=False) as episode:
                truth, valid = episode['target'][:, 0].copy(), episode['target_valid'][:, 0].copy()
            if not valid[eligible].all():
                raise ValueError('Saved score mask must retain original steering-label validity')
            original = predictions[key+'_original'][:, 0]
            delayed = predictions[key+'_delay_2_0'][:, 0]
            groups = {}
            for label, mask in (('lower_75_percent', low), ('upper_25_percent', high)):
                groups[label] = {'count': int(mask.sum()),
                    'change_min_median_max': [float(delta[mask].min()),
                        float(np.median(delta[mask])), float(delta[mask].max())],
                    'original_rmse': float(np.sqrt(np.mean((original[mask]-truth[mask])**2))),
                    'delay_2s_rmse': float(np.sqrt(np.mean((delayed[mask]-truth[mask])**2))),
                    'zero_rmse': float(np.sqrt(np.mean(truth[mask]**2))),
                    'mean_abs_output_difference': float(np.mean(np.abs(
                        original[mask]-delayed[mask])))}
            reports.append({'file': entry['file'], 'threshold': threshold, 'groups': groups})
            saved[key+'_change'], saved[key+'_high'], saved[key+'_eligible'] = delta, high, eligible
    np.savez_compressed(run/'strata.npz', **saved)
    result = {'status': 'completed_sensory_motion_change_strata_audit', 'reports': reports,
        'strata_sha256': sha256(run/'strata.npz'), 'same_track_known_development': True,
        'feature_change_is_not_physical_optic_flow_magnitude': True,
        'current_body_and_old_vision_is_counterfactual': True,
        'actor_deployed': False, 'optimizer_updates': 0, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
