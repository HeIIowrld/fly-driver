"""Diagnostic upstream-image versus neural readout, with fixed/learnable spatial pooling."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import activity, read
from scripts.train_spatial_preview_boundary import PreviewBoundary


class ColumnBoundary(nn.Module):
    def __init__(self, weight, mean, scale, learn_spatial):
        super().__init__()
        self.spatial = nn.Parameter(weight.clone(), requires_grad=learn_spatial)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.readout = PreviewBoundary(14)

    def forward(self, x):
        pooled = (x @ self.spatial.T).flatten(1)
        return self.readout(((pooled-self.mean)/self.scale).clamp(-5, 5))


def metrics(y, valid, clock, prediction, ym):
    report = {}
    for period in ('full', 'after3s'):
        heads = []
        for h in range(6):
            keep = valid[:, h].copy()
            if period == 'after3s':
                keep &= clock >= clock[0]+3
            actual, inferred = y[keep, h], prediction[keep, h]
            heads.append(dict(count=int(keep.sum()),
                rmse=float(np.sqrt(np.mean((actual-inferred)**2))),
                train_mean_rmse=float(np.sqrt(np.mean((actual-ym[h])**2))),
                zero_rmse=float(np.sqrt(np.mean(actual**2)))))
        report[period] = heads
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    conditions = ['intensity_fixed', 'intensity_learned', 'neural_fixed', 'neural_learned']
    if plan['conditions'] != conditions or plan['epochs'] != 200:
        raise ValueError('Preregistered four fixed-last200 conditions required')
    for file, digest in plan['source_sha256'].items():
        if sha256(file) != digest:
            raise ValueError('Executed source differs from preregistration')
    root = Path(plan['readout_batch'])
    batch = read(root/'result.json')
    source = read(root/'manifest.json')['config']
    if batch['status'] != 'completed_intermediate_visual_readout_batch':
        raise ValueError('Complete original neuron readout required')
    labels = read(Path(plan['preview_data'])/'dataset.json')
    targets = {e['id']: e for e in labels['episodes']}
    reference = Path(plan['reference_data'])
    manifest = read(reference/'dataset.json')
    calibration = reference/'channel-calibration.npz'
    if sha256(calibration) != manifest['channel_calibration_sha256']:
        raise ValueError('TRAIN channel calibration changed')
    with load_array(calibration, allow_pickle=False) as a:
        calibration_values = {k: a[k].copy() for k in a.files}
    episodes, provenance = [], []
    for entry in batch['episodes']:
        label = targets[entry['id']]
        if label['role'] not in ('train', 'development') or label['role'] != entry['role']:
            raise ValueError('No evaluation access or changed roles')
        data, clock = activity(entry, source)
        if sha256(label['targets_file']) != label['targets_sha256']:
            raise ValueError('Targets changed')
        with load_array(label['targets_file'], allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['observation_time'])
            y = a['preview'].reshape(-1, 6).copy()
            valid = np.repeat(a['preview_valid'], 2, axis=1)
        # Before-neuron intensity is the recorded image-derived retinal sample, not RGB CNN.
        raw = data['raw_control'][:, :, :1].copy()
        processed = data['processed_actual']
        raw = np.tanh((raw-calibration_values['raw_control_mean'][0])/
                      calibration_values['raw_control_scale'][0]).astype(np.float32)
        processed = np.tanh((processed-calibration_values['processed_actual_mean'][None,
            None, :, None])/calibration_values['processed_actual_scale'][None, None, :, None])
        episodes.append(dict(id=entry['id'], role=entry['role'], group=label['group'],
            raw=raw, neural=processed.astype(np.float32), y=y, valid=valid, clock=clock))
        provenance.append(dict(id=entry['id'], role=entry['role'],
            responses_sha256=entry['responses_sha256'], targets_sha256=label['targets_sha256']))
    train = [e for e in episodes if e['role'] == 'train']
    y, valid = (np.concatenate([e[k] for e in train]) for k in ('y', 'valid'))
    ym = np.array([y[valid[:, h], h].mean() for h in range(6)])
    ys = np.maximum([y[valid[:, h], h].std() for h in range(6)], plan['target_scale_floor'])
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training required')
    spec = read(plan['scene_spec'])
    if sha256(spec['scene_registration_file']) != spec['scene_registration_sha256']:
        raise ValueError('Original registered retinal geometry required')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as reg:
        uv = reg['image_uv'].copy()
    initial = FlightScenePool(uv, spec['scene_basis']).to('cuda').weight
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), original_histories=provenance,
        device=torch.cuda.get_device_name(0), actor_policy=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    batch_size = plan['batch_size']

    def inputs(e, condition, start=0, end=None):
        if condition.startswith('intensity'):
            value = torch.tensor(e['raw'][start:end], device='cuda')
            return torch.nn.functional.pad(value, (0, 0, 0, 13))
        return torch.tensor(e['neural'][start:end], device='cuda')

    records, reports = [], []
    for condition in conditions:
        # Fixed initial-pool moments use TRAIN only, before fitting either spatial/readout layer.
        pooled = []
        with torch.no_grad():
            for e in train:
                for i in range(0, len(e['y']), batch_size):
                    pooled.append((inputs(e, condition, i, i+batch_size) @ initial.T
                                   ).flatten(1).cpu().numpy())
        pooled = np.concatenate(pooled)
        mean = pooled.mean(0, dtype=float).astype(np.float32)
        scale = pooled.astype(float).std(0).clip(plan['feature_scale_floor']).astype(np.float32)
        torch.manual_seed(plan['seed'])
        model = ColumnBoundary(initial, torch.tensor(mean, device='cuda'),
            torch.tensor(scale, device='cuda'), condition.endswith('learned')).to('cuda')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
        losses = []
        counts = valid.sum(0)
        for epoch in range(plan['epochs']):
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for e in train:
                for i in range(0, len(e['y']), batch_size):
                    selection = slice(i, i+batch_size)
                    target = torch.tensor((e['y'][selection]-ym)/ys,
                                          device='cuda', dtype=torch.float32)
                    weight = torch.tensor(e['valid'][selection]/(6*counts),
                                          device='cuda', dtype=torch.float32)
                    loss = ((model(inputs(e, condition, i, i+batch_size))-target
                             ).square()*weight).sum()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                    total += float(loss.detach().cpu())
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.requires_grad):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            losses.append(total)
            if epoch % 10 == 0:
                write_json(run/'progress.json', dict(condition=condition, epoch=epoch+1,
                    loss=total, updates_per_epoch=1))
        model.eval()
        predictions = {}
        # Feature replacement occurs after spatial readout, using fixed TRAIN initial-pool mean.
        with torch.no_grad():
            mean_prediction = model.readout(torch.zeros((1, 896), device='cuda')
                                            ).cpu().numpy()[0]*ys+ym
            for e in episodes:
                pieces = [model(inputs(e, condition, i, i+batch_size)).cpu().numpy()*ys+ym
                          for i in range(0, len(e['y']), batch_size)]
                intact = np.concatenate(pieces)
                for intervention, prediction in [('intact', intact),
                        ('vision_train_initial_pool_mean',
                         np.broadcast_to(mean_prediction, intact.shape))]:
                    predictions[e['id']+'__'+intervention] = prediction
                    reports.append(dict(condition=condition, episode=e['id'], role=e['role'],
                        intervention=intervention,
                        **metrics(e['y'], e['valid'], e['clock'], prediction, ym)))
        np.savez_compressed(run/(condition+'-predictions.npz'), **predictions)
        torch.save(dict(model=model.state_dict(), plan=plan, condition=condition,
            target_mean=ym, target_scale=ys, channel_calibration_file=str(calibration),
            channel_calibration_sha256=sha256(calibration), actuator_policy=False),
            run/(condition+'.pt'))
        records.append(dict(condition=condition, losses=losses,
            maximum_spatial_weight_change=float((model.spatial-initial).abs().max().item())))
        write_json(run/'partial-result.json', dict(records=records, reports=reports))
    write_json(run/'result.json', dict(status='completed_column_preview_comparison',
        records=records, reports=reports, optimizer_updates=800,
        author_neuron_weights_changed=False, original_roles_and_clocks_preserved=True,
        target_clipping=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
