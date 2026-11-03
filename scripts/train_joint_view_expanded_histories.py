"""Train fixed four-view readout on two original role-preserving history corpora."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def prepare(entry, source, expected_uv=None):
    root = Path(source['output'])
    response, registration = root/'responses.npz', root/'retina-registration.npz'
    if (sha256(response) != source['responses_sha256']
            or sha256(entry['targets_file']) != entry['targets_sha256']):
        raise ValueError('Original role-bound response or target changed')
    pins = {str(p): sha256(p) for p in (response, registration, Path(entry['targets_file']))}
    with load_array(registration, allow_pickle=False) as data:
        uv = data['image_uv'].copy()
        if data['view_order'].tolist() != ['front', 'left', 'right', 'rear']:
            raise ValueError('Original four view ordering required')
    if uv.shape != (721, 2) or not np.isfinite(uv).all():
        raise ValueError('Original image coordinate geometry required')
    if expected_uv is not None and not np.array_equal(uv, expected_uv):
        raise ValueError('Inconsistent camera registration')
    extent = np.ptp(uv, axis=0)
    if np.any(extent <= 0):
        raise ValueError('Image-coordinate axes must have nonzero extent')
    cells = np.minimum(((uv-uv.min(axis=0))/extent*4).astype(int), 3)
    weights = np.zeros((721, 16), dtype=np.float64)
    for i in range(16):
        selected = (cells[:, 0] == i % 4) & (cells[:, 1] == i // 4)
        if selected.any():
            weights[selected, i] = 1/selected.sum()
    with load_array(response, allow_pickle=False) as data:
        if (data['additional_cell_types'].tolist() != ['L1', 'L2', 'Mi1', 'Tm3', 'Tm1', 'Tm9']
                or data['motion_activity'].shape != (entry['observations'], 4, 8, 721)):
            raise ValueError('Original author retinal and motion activities required')
        clock = data['decision_monotonic_seconds'].copy()
        retinal = np.asarray(data['additional_activity']) @ weights
        motion = np.asarray(data['motion_activity']) @ weights
    with load_array(entry['targets_file'], allow_pickle=False) as data:
        np.testing.assert_array_equal(clock, data['observation_time'])
        y = data['preview'].reshape(-1, 6)[:, 5].copy()
        valid = data['preview_valid'][:, 2].copy()
    n = entry['observations']
    x = np.concatenate([retinal.reshape(n, -1), motion.reshape(n, -1)], axis=1)
    if x.shape != (n, 896) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Complete finite original histories required')
    return {'id': entry['id'], 'role': entry['role'], 'x': x, 'y': y, 'valid': valid,
            'uv': uv, 'pins': pins}


def score(e, pred, mean):
    mask = e['valid']
    y, p = e['y'][mask], pred[mask]
    groups = {}
    for name, selected in [('negative', y < -.001), ('positive', y > .001)]:
        groups[name] = {'rows': int(selected.sum()), 'same_sign_fraction': (
            float(np.mean(np.sign(y[selected]) == np.sign(p[selected])))
            if selected.any() else None)}
    return {'valid_rows': int(mask.sum()),
        'rmse_per_m': float(np.sqrt(np.mean((p-y)**2))),
        'constant_train_mean_rmse_per_m': float(np.sqrt(np.mean((mean-y)**2))),
        'direction_groups': groups}


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
    old_sources = read(args.older_readout)['episodes']
    new_sources = read(plan['sensor_replay'])['episodes']
    sources = {e['id']: e for e in old_sources + new_sources}
    if len(sources) != len(entries) or len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Unique role-bound history IDs required')
    recordings = [str(Path(sources[e['id']]['recording']).resolve()) for e in entries]
    if len(set(recordings)) != len(recordings):
        raise ValueError('Repeated original recording must not be duplicated')
    for e in entries:
        if e['role'] != sources[e['id']]['role']:
            raise ValueError('Original TRAIN/development role must remain unchanged')
    train = [e for e in entries if e['role'] == 'train']
    development = [e for e in entries if e['role'] == 'development']
    if len(train) != 11 or len(development) != 3:
        raise ValueError('Expected 8 old and 3 new TRAIN, 2 old and 1 new DEV')
    torch.set_num_threads(1)
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_plan_sha256': args.plan_sha256, 'train_ids': [e['id'] for e in train],
        'older_dataset_sha256': sha256(args.older_dataset),
        'older_readout_sha256': sha256(args.older_readout),
        'model': '896 -> tanh 64 -> 20m curvature', 'seed': 863692,
        'updates': 200, 'learning_rate': .001, 'weight_decay': .01,
        'development_read_before_checkpoint': False, 'actor_deployed': False})
    records = []
    for e in train:
        records.append(prepare(e, sources[e['id']]))
        print(json.dumps({'loaded_train': e['id']}), flush=True)
    for r in records[1:]:
        np.testing.assert_array_equal(records[0]['uv'], r['uv'])
    x = np.concatenate([r['x'] for r in records])
    y = np.concatenate([r['y'] for r in records])
    valid = np.concatenate([r['valid'] for r in records])
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-5)
    target_mean, target_scale = float(y[valid].mean()), max(float(y[valid].std()), 1e-4)
    tx = torch.tensor(np.clip((x-mean)/scale, -5, 5), dtype=torch.float32)
    ty = torch.tensor((y-target_mean)/target_scale, dtype=torch.float32)
    mask = torch.tensor(valid)
    torch.manual_seed(863692)
    model = nn.Sequential(nn.Linear(896, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    losses = []
    for i in range(200):
        opt.zero_grad(set_to_none=True)
        loss = (model(tx).squeeze(1)[mask]-ty[mask]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        opt.step()
        if i in (0, 199):
            losses.append(float(loss.detach()))
    model.eval()
    checkpoint = run/'joint-view-preview.pt'
    torch.save({'model': model.state_dict(), 'feature_mean': mean, 'feature_scale': scale,
        'target_mean': target_mean, 'target_scale': target_scale,
        'train_ids': [r['id'] for r in records], 'actor_qualified': False,
        'author_network_weights_changed': False}, checkpoint)
    print(json.dumps({'stage': 'checkpoint_fixed', 'updates': 200}), flush=True)
    # Previously known development is used for evaluation only, not a blind final test.
    dev = [prepare(e, sources[e['id']], records[0]['uv']) for e in development]
    reports = []
    for e in [*records, *dev]:
        normalized = torch.tensor(np.clip((e['x']-mean)/scale, -5, 5), dtype=torch.float32)
        with torch.no_grad():
            prediction = model(normalized).squeeze(1).numpy()*target_scale + target_mean
        reports.append({'id': e['id'], 'role': e['role'],
                        'metrics': score(e, prediction, target_mean)})
    for e in [*records, *dev]:
        if any(sha256(p) != h for p, h in e['pins'].items()):
            raise ValueError('Input changed during experiment')
    write_json(run/'result.json', {'status': 'completed_fixed_joint_view_readout',
        'reports': reports, 'train_loss_first_last_preupdate': losses,
        'recordings_unique_by_resolved_path': True,
        'optimizer_updates': 200, 'author_network_optimizer_updates': 0,
        'checkpoint_sha256': sha256(checkpoint), 'development_read_before_checkpoint': False,
        'known_development_is_not_independent_final_test': True,
        'target': '20m teacher path chord curvature, not corner distance or optimal line',
        'visual_features': 'four views, 6 retinal and 8 motion types at 16 image bins each',
        'body_or_map_input_at_readout': False, 'actor_deployed': False,
        'graph_and_real_driving_evaluated': False, 'goal_achieved': False})
    for row in reports:
        if row['role'] == 'development':
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
