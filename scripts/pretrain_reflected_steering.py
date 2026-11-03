"""Visual command/preview pretraining, without bypassing the fly motor circuit in a driver."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


class VisualCommandPretrain(nn.Module):
    """Engineering higher-visual boundary; transferable latent64, not an actuator policy."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(9, 8)
        self.hidden = nn.Linear(512, 64)
        self.training_outputs = nn.Linear(64, 4)

    def forward(self, x):
        patches = x.reshape(-1, 4, 9, 16).transpose(2, 3)
        latent = self.patch(patches).tanh().flatten(1)
        return self.training_outputs(self.hidden(latent).tanh())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root, preview, mirrored = (Path(plan[k]) for k in ('data', 'preview', 'reflected'))
    replay = read(mirrored/'result.json')
    assert replay['status'] == 'completed_mirrored_author_visual_replay'
    assert replay['author_weights_unchanged'] and replay['original_baseline_max_error'] <= 2e-6
    mirror_entries = {e['file']: e for e in replay['episodes']}
    preview_entries = {e['file']: e for e in read(preview/'result.json')['episodes']}
    episodes = []
    for entry in read(root/'dataset.json')['episodes']:
        assert entry['role'] in ('train', 'development')
        assert sha256(root/entry['file']) == entry['sha256']
        assert sha256(preview/entry['file']) == preview_entries[entry['file']]['auxiliary_sha256']
        with load_array(root/entry['file'], allow_pickle=False) as a:
            x, y, valid, clock = (a[k].copy() for k in
                ('scene_features', 'target', 'target_valid', 'observation_time'))
        with load_array(preview/entry['file'], allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['observation_time'])
            targets = np.column_stack([y[:, 0], a['preview'][:, :, 0]])
            masks = np.column_stack([valid[:, 0], a['preview_valid']])
        e = dict(file=entry['file'], role=entry['role'], x=x, y=targets, valid=masks, clock=clock)
        if entry['role'] == 'train':
            assert sha256(mirrored/entry['file']) == mirror_entries[entry['file']]['sha256']
            assert mirror_entries[entry['file']]['original_episode_sha256'] == entry['sha256']
            with load_array(mirrored/entry['file'], allow_pickle=False) as a:
                np.testing.assert_array_equal(clock, a['observation_time'])
                e['mirror_x'] = a['scene_features'].copy()
        episodes.append(e)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), mirrored_result_sha256=sha256(mirrored/'result.json'),
        parent_dataset_sha256=sha256(root/'dataset.json'),
        preview_result_sha256=sha256(preview/'result.json'),
        development_used_only_for_terminal_reporting=True, evaluation_arrays_opened=False,
        source_qualification_upgraded=False, device=torch.cuda.get_device_name(0)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'plan.json').write_bytes(Path(args.plan).read_bytes())
    original = [e for e in episodes if e['role'] == 'train']
    physical_y = np.concatenate([e['y'] for e in original])
    physical_mask = np.concatenate([e['valid'] for e in original])
    baseline = np.array([physical_y[physical_mask[:, h], h].mean() for h in range(4)])
    reports, records = [], []
    for condition in plan['conditions']:
        arrays, targets, masks = [], [], []
        for e in original:
            arrays.append(e['x'])
            targets.append(e['y'])
            masks.append(e['valid'])
            if condition == 'original_plus_reflected':
                arrays.append(e['mirror_x'])
                targets.append(-e['y'])
                masks.append(e['valid'])
        x, y, valid = np.concatenate(arrays), np.concatenate(targets), np.concatenate(masks)
        mean = x.mean(0, dtype=float)
        scale = x.astype(float).std(0).clip(plan['feature_scale_floor'])
        target_mean = np.array([y[valid[:, h], h].mean() for h in range(4)])
        target_scale = np.maximum([y[valid[:, h], h].std() for h in range(4)],
                                  plan['target_scale_floor'])
        groups = np.where(y[:, 0] > .01, 0, np.where(y[:, 0] < -.01, 1, 2))
        counts = np.bincount(groups[valid[:, 0]], minlength=3)
        assert (counts > 0).all()
        weight = np.zeros_like(y)
        weight[:, 0] = .5*valid[:, 0]/(3*counts[groups])
        weight[:, 1:] = .5*valid[:, 1:]/(3*valid[:, 1:].sum(0))
        batch = torch.tensor(((x-mean)/scale).clip(-5, 5), device='cuda', dtype=torch.float32)
        target = torch.tensor((y-target_mean)/target_scale, device='cuda', dtype=torch.float32)
        weights = torch.tensor(weight, device='cuda', dtype=torch.float32)
        torch.manual_seed(plan['seed'])
        model = VisualCommandPretrain().to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['weight_decay'])
        losses = []
        for epoch in range(plan['epochs']):
            optimizer.zero_grad(set_to_none=True)
            loss = (((model(batch)-target)**2)*weights).sum()
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if epoch % 25 == 0:
                write_json(run/'progress.json', dict(condition=condition, epoch=epoch+1,
                                                     loss=losses[-1]))
        torch.save(dict(schema='flybrain.visual.command_pretrain.v1', model=model.state_dict(),
            feature_mean=mean, feature_scale=scale, target_mean=target_mean,
            target_scale=target_scale, condition=condition, plan=plan,
            actuator_policy=False), run/(condition+'.pt'))
        predictions = {}
        model.eval()
        for e in episodes:
            with torch.no_grad():
                xx = torch.tensor(((e['x']-mean)/scale).clip(-5, 5),
                                  device='cuda', dtype=torch.float32)
                prediction = model(xx).cpu().numpy()*target_scale+target_mean
            predictions[e['file']] = prediction
            report = dict(condition=condition, file=e['file'], role=e['role'])
            for period in ('full', 'after3s'):
                rows = []
                for h in range(4):
                    keep = e['valid'][:, h].copy()
                    if period == 'after3s':
                        keep &= e['clock'] >= e['clock'][0]+3
                    yy, pp = e['y'][keep, h], prediction[keep, h]
                    rows.append(dict(count=int(keep.sum()),
                        rmse=float(np.sqrt(((pp-yy)**2).mean())),
                        original_train_mean_rmse=float(np.sqrt(((baseline[h]-yy)**2).mean())),
                        zero_rmse=float(np.sqrt((yy**2).mean()))))
                report[period] = rows
            reports.append(report)
        np.savez_compressed(run/(condition+'-predictions.npz'), **predictions)
        records.append(dict(condition=condition, epochs=len(losses), losses=losses,
                            fit_rows=len(y), group_counts=counts.tolist()))
    write_json(run/'result.json', dict(status='completed_fixed_visual_steering_pretraining',
        records=records, reports=reports, author_visual_checkpoint_unchanged=True,
        source_roles_labels_clocks_unchanged=True, fitted_actor_inputs='scene576 only',
        actuator_policy=False, flight_motor_transfer_performed=False,
        physiology_or_racing_claim=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
