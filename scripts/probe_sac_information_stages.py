"""Compare spatial/neural-stage decodability with fixed train-only probes.

These probes never replace the measured flight circuit or issue vehicle controls.
All development is the consumed historical Monza episode. Stage PCA is learned
on TRAIN neural states only and is an analysis tool, not a neuronal claim.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain_drive.anticipation_targets import (diagnostic_features,
    future_command_targets, ridge_fit, ridge_predict)
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.diagnose_sac_anticipation import load_verified, metric, read


def train_pca(train, count=128, seed=780403):
    x = np.concatenate(train).astype(np.float32)
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    scale = x.std(0, dtype=np.float64).astype(np.float32)
    active = scale > 1e-8
    z = (x[:, active] - mean[active]) / scale[active]
    q = min(count+8, z.shape[0]-1, z.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((z.shape[1], q)).astype(np.float32)
    y = z @ omega
    for _ in range(2):
        y, _ = np.linalg.qr(y, mode='reduced')
        y = z @ (z.T @ y)
    y, _ = np.linalg.qr(y, mode='reduced')
    _, _, vt = np.linalg.svd(y.T @ z, full_matrices=False)
    components = vt[:min(count, len(vt))].T.copy()
    explained = float(np.square(z @ components).sum(dtype=np.float64) / np.square(z).sum(dtype=np.float64))
    return dict(mean=mean, scale=scale, active=active, components=components,
                standardized_variance_fraction=explained)


def apply_pca(pca, x):
    return ((x[:, pca['active']] - pca['mean'][pca['active']]) /
            pca['scale'][pca['active']]) @ pca['components']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['spatial', 'neural'], required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    entries = read('runs/teacher-synchronized-20261102/student-split-01.json')['episodes']
    data = [load_verified(e) for e in entries]
    source = Path(a.source)
    sr = read(source / 'result.json')
    expected = 'completed_qualified_static_spatial_recovery' if a.mode == 'spatial' else 'completed_premuscle_state_trace'
    if sr['status'] != expected:
        raise ValueError('Completed exact source evidence required')
    source_files = sr.get('files_sha256')
    for name, expected_hash in source_files.items():
        if sha256(source / name) != expected_hash:
            raise ValueError('Source arrays changed')
    run = begin_run(a.output, dict(arguments=vars(a), seed=780403, program_sha256=sha256(__file__),
        entries=entries, source_result_sha256=sha256(source/'result.json'),
        scope='Consumed development only, fixed ridge .01, train-only normalization/PCA',
        horizons=[0, .25, .5, 1, 2], train_only_selection_folds='Whole episode 0->1 and 1->0, horizon0/1s',
        neuronal_state_timing='States integrate prior held drive; visual_base/current drive use the newly available current scene',
        optimizer_updates=0, actor_deployed=False))
    result = dict(status='failed', biological_optimizer_updates=0, vehicle_control_sent=False,
                  actor_deployed=False, probe_fits=0)
    try:
        features = []
        pcas = {}
        if a.mode == 'neural':
            groups = read(source/'groups.json')
            states = [np.load(source/f'state-{i}.npy', mmap_mode='r', allow_pickle=False) for i in range(3)]
            for group in ['premotor', 'full']:
                indices = groups['premotor'] if group == 'premotor' else np.arange(states[0].shape[1])
                pcas[group] = train_pca([s[:, indices] for s in states[:2]])
                np.savez_compressed(run/f'{group}-pca.npz', **pcas[group])
            for i, (d, s) in enumerate(zip(data, states)):
                ix = np.flatnonzero(d['fit_mask'])
                base = diagnostic_features(d)
                f = {name: base[name][ix] for name in ['body', 'vision', 'time_only']}
                with np.load(source/f'observed-{i}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['event_indices'], ix)
                    for name in ['visual_base', 'visual_drive', 'muscles']:
                        f[name] = z[name].copy()
                for name in ['vision', 'body', 'steering', 'goal', 'motor']:
                    f['state_'+name] = np.asarray(s[:, groups[name]])
                f['state_descending'] = np.concatenate([f['state_steering'], f['state_goal']], axis=1)
                f['premotor_pca128'] = apply_pca(pcas['premotor'], s[:, groups['premotor']])
                f['full_pca128'] = apply_pca(pcas['full'], s)
                f['visual_and_descending'] = np.concatenate([f['state_vision'], f['state_descending']], axis=1)
                features.append(f)
        else:
            for i, d in enumerate(data):
                ix = np.flatnonzero(d['fit_mask'])
                base = diagnostic_features(d)
                f = {name: base[name][ix] for name in ['body', 'vision', 'time_only', 'intensity']}
                with np.load(source/f'spatial-{i}.npz', allow_pickle=False) as z:
                    joined = z['event_to_camera'][ix]
                    np.testing.assert_array_equal(z['camera_indices'][joined], d['camera_index'][ix])
                    for name in ['grid4_wide', 'grid8_wide', 'grid4_narrow', 'grid8_narrow', 'discarded_grid8']:
                        f[name] = z[name][joined].copy()
                for name in ['discarded_grid8', 'grid8_narrow']:
                    f['vision_plus_'+name] = np.concatenate([f['vision'], f[name]], axis=1)
                    f['body_vision_plus_'+name] = np.concatenate([f['body'], f['vision'], f[name]], axis=1)
                f['body_vision'] = np.concatenate([f['body'], f['vision']], axis=1)
                features.append(f)
        targets, masks = [], []
        for d in data:
            ix = np.flatnonzero(d['fit_mask'])
            future, fm, _ = future_command_targets(d['decision_time'], d['labels'], d['label_valid'], d['fit_mask'])
            targets.append([d['labels'][ix], *[future[ix, h] for h in range(4)]])
            masks.append([np.ones(len(ix), dtype=bool), *[fm[ix, h] for h in range(4)]])
        saved, reports, folds = {}, [], []
        for h, horizon in enumerate([0, .25, .5, 1., 2.]):
            report = dict(horizon_seconds=horizon, probes={})
            y = np.concatenate([t[h][m[h]] for t, m in zip(targets[:2], masks[:2])])
            dev_y = targets[2][h][masks[2][h]]
            saved[f'h{h}_target'] = dev_y
            for name in features[0]:
                x = np.concatenate([f[name][m[h]] for f, m in zip(features[:2], masks[:2])])
                fitted = ridge_fit(x, y, .01)
                pred = ridge_predict(fitted, features[2][name][masks[2][h]])
                report['probes'][name] = metric(pred, dev_y)
                saved[f'h{h}_{name}_prediction'] = pred
                result['probe_fits'] += 1
                if h in [0, 3] and a.mode == 'spatial':
                    for tr, val in [(0, 1), (1, 0)]:
                        fitted_fold = ridge_fit(features[tr][name][masks[tr][h]], targets[tr][h][masks[tr][h]], .01)
                        pred_fold = ridge_predict(fitted_fold, features[val][name][masks[val][h]])
                        folds.append(dict(horizon_seconds=horizon, feature=name, train_episode=tr,
                            val_episode=val, **metric(pred_fold, targets[val][h][masks[val][h]])))
                        result['probe_fits'] += 1
                write_json(run/'progress.json', dict(horizon=horizon, feature=name, probe_fits=result['probe_fits']))
            reports.append(report)
        np.savez_compressed(run/'predictions.npz', **saved)
        write_json(run/'train-only-folds.json', folds)
        result.update(status='completed_information_stage_probes', reports=reports,
            feature_dimensions={k: v.shape[1] for k, v in features[0].items()},
            pca_variance={k: p['standardized_variance_fraction'] for k, p in pcas.items()},
            predictions_sha256=sha256(run/'predictions.npz'), train_folds_sha256=sha256(run/'train-only-folds.json'),
            interpretation='Information decodability, not physiological causality or autonomous driving. PCA cannot prove absence of information in full hidden state.',
            termination_reason='Fixed stage representations and forecast horizons compared')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit probe failure')
    write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
