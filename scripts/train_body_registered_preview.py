"""Matched image-grid/body-registered readouts; no control policy or anatomical weight updates."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_column_preview_comparison import metrics
from scripts.train_visual_body_preview import body_features, body_moments, direct_body_reference


class RegisteredPreview(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(1024, 128)
        self.body = nn.Linear(42, 16)
        self.hidden = nn.Linear(144, 64)
        self.output = nn.Linear(64, 6)

    def forward(self, vision, body):
        combined = torch.cat([self.vision(vision).tanh(), self.body(body).tanh()], dim=1)
        return self.output(self.hidden(combined).tanh())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = read(a.plan)
    if (plan['conditions'] != ['image_grid', 'registered', 'registered_level']
            or plan['epochs'] != 200 or plan['body_route_indices'] != [0, 1, 4, 16]):
        raise ValueError('Fixed representation comparison and geometry-only body routes required')
    for file, digest in plan['source_sha256'].items():
        if sha256(file) != digest:
            raise ValueError('Preregistered training source changed')
    root = Path(plan['data'])
    source = read(root/'dataset.json')
    if (source['schema'] != 'flybrain.body_registered_preview.v1'
            or sorted(e['role'] for e in source['episodes']) != [
                'development', 'train', 'train', 'train']):
        raise ValueError('All four original new-mount histories required')
    episodes = []
    for entry in source['episodes']:
        for key in ('features', 'targets'):
            if sha256(entry[key+'_file']) != entry[key+'_sha256']:
                raise ValueError('Whole original inputs/labels changed')
        with load_array(entry['features_file'], allow_pickle=False) as x, load_array(
                entry['targets_file'], allow_pickle=False) as y:
            clock = x['observation_time'].copy()
            np.testing.assert_array_equal(clock, y['observation_time'])
            features = {k: x[k].copy() for k in x.files if k != 'observation_time'}
            target, valid = y['preview'].reshape(-1, 6).copy(), np.repeat(y['preview_valid'], 2, 1)
        body, mask, _ = direct_body_reference(entry, clock, sha256(plan['body_contract']))
        episodes.append(dict(entry, features=features, y=target, valid=valid, clock=clock,
                             body=body, body_valid=mask))
    train = [e for e in episodes if e['role'] == 'train']
    y, valid = (np.concatenate([e[k] for e in train]) for k in ('y', 'valid'))
    ym = np.array([y[valid[:, h], h].mean() for h in range(6)])
    ys = np.maximum([y[valid[:, h], h].std() for h in range(6)], plan['target_scale_floor'])
    bm, bs, counts = body_moments(episodes)
    for e in episodes:
        e['b'] = body_features(e['body'], e['body_valid'], bm, bs, plan['body_route_indices'])
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    run = begin_run(a.output, dict(plan=plan, plan_sha256=sha256(a.plan),
        program_sha256=sha256(__file__), dataset_sha256=sha256(root/'dataset.json'),
        body_train_valid_counts=counts.tolist(), development_used_to_fit=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    ty = torch.tensor((y-ym)/ys, device='cuda', dtype=torch.float32)
    tw = torch.tensor(valid/(6*valid.sum(0)), device='cuda', dtype=torch.float32)
    tb = torch.tensor(np.concatenate([e['b'] for e in train]), device='cuda')
    records, reports = [], []
    for condition in plan['conditions']:
        x = np.concatenate([e['features'][condition] for e in train])
        mean, scale = x.mean(0, dtype=float), x.astype(float).std(0).clip(1e-5)
        # Support bits are explicit indicators, not tiny-variance analog signals.
        mean[896:], scale[896:] = 0., 1.
        normalized = ((x-mean)/scale).clip(-5, 5).astype(np.float32)
        tx = torch.tensor(normalized, device='cuda')
        torch.manual_seed(plan['seed'])
        model = RegisteredPreview().to('cuda')
        initial = hashlib.sha256()
        for name, parameter in model.named_parameters():
            initial.update(name.encode('utf8'))
            initial.update(parameter.detach().cpu().numpy().tobytes())
        initial_sha = initial.hexdigest()
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['weight_decay'])
        losses = []
        for epoch in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx, tb)-ty).square()*tw).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite registered readout loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite registered readout gradient')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if epoch % 25 == 0:
                write_json(run/'progress.json', dict(condition=condition, epoch=epoch+1,
                                                     loss=losses[-1]))
        model.eval()
        predictions = {}
        for e in episodes:
            base = ((e['features'][condition]-mean)/scale).clip(-5, 5).astype(np.float32)
            variants = dict(intact=base)
            changed = base.copy()
            changed[:, :896] = 0  # Retain body-derived support patterns.
            variants['neural_feature_mean'] = changed
            raw = e['features'][condition].copy()
            raw[:, :896] = 0
            variants['retinal_neural_mean'] = ((raw-mean)/scale).clip(-5, 5).astype(np.float32)
            for view in ('front', 'left', 'right', 'rear'):
                key = condition+'_'+view+'_mean'
                if key in e['features']:
                    variants[view+'_neural_mean'] = ((e['features'][key]-mean)/scale).clip(
                        -5, 5).astype(np.float32)
            body = torch.tensor(e['b'], device='cuda')
            for intervention, values in variants.items():
                with torch.no_grad():
                    inferred = model(torch.tensor(values, device='cuda'), body).cpu().numpy()*ys+ym
                predictions[e['id']+'__'+intervention] = inferred
                reports.append(dict(condition=condition, episode=e['id'], role=e['role'],
                    intervention=intervention,
                    **metrics(e['y'], e['valid'], e['clock'], inferred, ym)))
        np.savez_compressed(run/(condition+'-predictions.npz'), **predictions)
        torch.save(dict(model=model.state_dict(), plan=plan, condition=condition, feature_mean=mean,
            feature_scale=scale, body_mean=bm, body_scale=bs, target_mean=ym, target_scale=ys,
            actuator_policy=False), run/(condition+'.pt'))
        records.append(dict(condition=condition, epochs=200, losses=losses,
            initial_parameter_sha256=initial_sha,
            parameter_count=sum(p.numel() for p in model.parameters())))
    write_json(run/'result.json', dict(status='completed_body_registered_preview_learning',
        records=records, reports=reports, optimizer_updates=600, train_target_scale=ys.tolist(),
        support_masks_preserved_in_neural_controls=True, author_weights_changed=False,
        flight_motor_transfer=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
