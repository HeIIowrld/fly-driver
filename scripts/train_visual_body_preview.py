"""Matched visual/body auxiliary preview learning, without privileged actor inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_column_preview_comparison import metrics


def body_moments(episodes):
    rows = [e for e in episodes if e['role'] == 'train']
    if not rows:
        raise ValueError('TRAIN histories required')
    values, valid = (np.concatenate([e[k] for e in rows]) for k in ('body', 'body_valid'))
    if values.shape != valid.shape or values.shape[1] != 21 or not np.isfinite(values).all():
        raise ValueError('Registered21 and explicit validity required')
    mean, scale = np.zeros(21), np.ones(21)
    for h in range(21):
        if valid[:, h].any():
            actual = values[valid[:, h], h]
            mean[h], scale[h] = actual.mean(), max(actual.std(), 1e-5)
    return mean, scale, valid.sum(0)


def direct_body_reference(entry, clock, contract_sha):
    root = Path(entry['body_replay'])
    proof, path = read(root/'result.json'), root/'virtual-body.npz'
    if (entry['role'] not in ('train', 'development')
            or sha256(root/'result.json') != entry['body_result_sha256']
            or proof['status'] != 'completed' or not proof['legacy16_whole_replay_exact']
            or proof['contract_sha256'] != contract_sha
            or sha256(path) != proof['virtual_body_sha256']):
        raise ValueError('Exact original fixed-role body21 reference required')
    with load_array(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(clock, arrays['decision_monotonic_seconds'])
        values, valid = arrays['values'].copy(), arrays['valid'].copy()
    if (values.shape != (len(clock), 21) or valid.shape != values.shape
            or valid.dtype != bool or not np.isfinite(values).all()):
        raise ValueError('Complete body21 with original explicit validity required')
    return values, valid, path


def body_features(values, valid, mean, scale, route_indices=None):
    """Head-specific routing; never edit the recorded values or sensory validity."""
    values, valid = np.asarray(values), np.asarray(valid)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if (values.ndim != 2 or values.shape[1] != 21 or valid.shape != values.shape
            or valid.dtype != bool or mean.shape != (21,) or scale.shape != (21,)
            or not np.isfinite(values).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError('Finite body21 and original validity/calibration required')
    indices = list(range(21)) if route_indices is None else route_indices
    if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 21 for i in indices)):
        raise ValueError('Distinct explicitly routed body channel indices required')
    route = np.zeros(21, bool)
    route[indices] = True
    available = valid & route[None]
    standardized = np.where(available, ((values-mean)/scale).clip(-5, 5), 0)
    return np.concatenate([standardized, available], axis=1).astype(np.float32)


class FusedPreview(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(14, 8)
        self.body = nn.Linear(42, 16)
        self.hidden = nn.Linear(528, 64)
        self.output = nn.Linear(64, 6)

    def forward(self, vision, body):
        spatial = self.patch(vision.reshape(-1, 4, 14, 16).transpose(2, 3)).tanh().flatten(1)
        fused = torch.cat([spatial, self.body(body).tanh()], dim=1)
        return self.output(self.hidden(fused).tanh())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if plan['conditions'] != ['vision_only', 'body_only', 'vision_body'] or plan['epochs'] != 200:
        raise ValueError('Preregistered matched conditions and fixed last200 required')
    for file, digest in plan['source_sha256'].items():
        if sha256(file) != digest:
            raise ValueError('Preregistered source changed')
    root = Path(plan['visual_data'])
    source = read(root/'dataset.json')
    if source['schema'] != 'flybrain.retinotopic_preview.episodes.v1':
        raise ValueError('Actual registered14 visual dataset required')
    body_contract_sha = sha256(plan['body_contract'])
    direct = plan.get('body_source') == 'audited_episode_references'
    if direct:
        if source.get('camera_mount_contract') != read(plan['camera_config'])[
                'camera_mount_contract']:
            raise ValueError('Explicit new-mount sensor references required')
        old = {}
    else:
        old_root = Path(plan['original_body_data'])
        old_manifest = read(old_root/'dataset.json')
        if old_manifest['body_contract_sha256'] != body_contract_sha:
            raise ValueError('Identical explicit21 body contract required')
        old = {'original-'+e['file'].removesuffix('.npz'): e for e in old_manifest['episodes']}
    episodes, provenance = [], []
    for e in source['episodes']:
        if e['role'] not in ('train', 'development'):
            raise ValueError('No evaluation access')
        for name in ('features', 'targets'):
            if sha256(e[name+'_file']) != e[name+'_sha256']:
                raise ValueError('Whole visual/target source changed')
        with load_array(e['features_file'], allow_pickle=False) as v, load_array(
                e['targets_file'], allow_pickle=False) as t:
            np.testing.assert_array_equal(v['observation_time'], t['observation_time'])
            vision, clock = v['processed_actual'].copy(), v['observation_time'].copy()
            y, valid = t['preview'].reshape(-1, 6).copy(), np.repeat(t['preview_valid'], 2, axis=1)
        if direct:
            body, body_valid, path = direct_body_reference(e, clock, body_contract_sha)
        elif e['id'] in old:
            original = old[e['id']]
            path = old_root/original['file']
            if original['role'] != e['role'] or sha256(path) != original['sha256']:
                raise ValueError('Original body source/role changed')
            with load_array(path, allow_pickle=False) as b:
                np.testing.assert_array_equal(clock, b['observation_time'])
                body, body_valid = b['body'].copy(), b['body_valid'].copy()
        else:
            body_root = Path(plan['new_body_data'])/e['id'].removeprefix('spatial-')
            proof = read(body_root/'result.json')
            path = body_root/'virtual-body.npz'
            if (e['role'] != 'train' or proof['status'] != 'completed'
                    or not proof['legacy16_whole_replay_exact']
                    or proof['contract_sha256'] != body_contract_sha
                    or sha256(path) != proof['virtual_body_sha256']):
                raise ValueError('Whole TRAIN21 replay with exact legacy preservation required')
            with load_array(path, allow_pickle=False) as b:
                np.testing.assert_array_equal(clock, b['decision_monotonic_seconds'])
                body, body_valid = b['values'].copy(), b['valid'].copy()
        episodes.append(dict(id=e['id'], role=e['role'], vision=vision, clock=clock,
                             y=y, valid=valid, body=body, body_valid=body_valid))
        provenance.append(dict(id=e['id'], role=e['role'], body_file=str(path),
            body_sha256=sha256(path), visual_sha256=e['features_sha256'],
            targets_sha256=e['targets_sha256']))
    train = [e for e in episodes if e['role'] == 'train']
    vision, y, valid = (np.concatenate([e[k] for e in train]) for k in ('vision', 'y', 'valid'))
    vm, vs = vision.mean(0, dtype=float), vision.astype(float).std(0).clip(1e-5)
    bm, bs, body_counts = body_moments(episodes)
    ym = np.array([y[valid[:, h], h].mean() for h in range(6)])
    ys = np.maximum([y[valid[:, h], h].std() for h in range(6)], plan['target_scale_floor'])
    for e in episodes:
        e['v'] = ((e['vision']-vm)/vs).clip(-5, 5).astype(np.float32)
        e['b'] = body_features(e['body'], e['body_valid'], bm, bs, plan.get('body_route_indices'))
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        source_sha256=sha256(__file__), original_sources=provenance,
        body_train_valid_counts=body_counts.tolist(), device=torch.cuda.get_device_name(0)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    tx = torch.tensor(np.concatenate([e['v'] for e in train]), device='cuda')
    tb = torch.tensor(np.concatenate([e['b'] for e in train]), device='cuda')
    ty = torch.tensor((y-ym)/ys, device='cuda', dtype=torch.float32)
    tw = torch.tensor(valid/(6*valid.sum(0)), device='cuda', dtype=torch.float32)
    records, reports = [], []
    for condition in plan['conditions']:
        torch.manual_seed(plan['seed'])
        model = FusedPreview().to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['weight_decay'])
        x = torch.zeros_like(tx) if condition == 'body_only' else tx
        b = torch.zeros_like(tb) if condition == 'vision_only' else tb
        losses = []
        for epoch in range(plan['epochs']):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x, b)-ty).square()*tw).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite joint gradient')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if epoch % 25 == 0:
                write_json(run/'progress.json', dict(condition=condition, epoch=epoch+1,
                                                     loss=losses[-1]))
        predictions = {}
        with torch.no_grad():
            for e in episodes:
                vx = torch.tensor(e['v'], device='cuda')
                bx = torch.tensor(e['b'], device='cuda')
                if condition == 'body_only':
                    vx.zero_()
                if condition == 'vision_only':
                    bx.zero_()
                interventions = [('intact', vx, bx),
                        ('vision_absent', torch.zeros_like(vx), bx),
                        ('body_absent', vx, torch.zeros_like(bx))]
                for name, indices in plan.get('body_channel_interventions', {}).items():
                    if not indices or any(type(i) is not int or not 0 <= i < 21 for i in indices):
                        raise ValueError('Explicit original body channel indices required')
                    changed = bx.clone()
                    changed[:, indices] = 0
                    changed[:, [i+21 for i in indices]] = 0
                    interventions.append((name, vx, changed))
                for intervention, vi, bi in interventions:
                    p = model(vi, bi).cpu().numpy()*ys+ym
                    predictions[e['id']+'__'+intervention] = p
                    reports.append(dict(condition=condition, episode=e['id'], role=e['role'],
                        intervention=intervention,
                        **metrics(e['y'], e['valid'], e['clock'], p, ym)))
        np.savez_compressed(run/(condition+'-predictions.npz'), **predictions)
        torch.save(dict(model=model.state_dict(), plan=plan, condition=condition,
            vision_mean=vm, vision_scale=vs, body_mean=bm, body_scale=bs,
            target_mean=ym, target_scale=ys, actuator_policy=False), run/(condition+'.pt'))
        records.append(dict(condition=condition, losses=losses))
    write_json(run/'result.json', dict(status='completed_visual_body_preview_comparison',
        records=records, reports=reports, optimizer_updates=600,
        body_train_valid_counts=body_counts.tolist(), author_weights_changed=False,
        original_roles_clocks_masks_preserved=True, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
