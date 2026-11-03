"""Paired fixed visual-boundary learning; future paths remain targets, never actor inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_preview_learning import training_moments


class PreviewBoundary(nn.Module):
    """Engineering higher-visual adapter; no direct motor/actuator policy."""

    def __init__(self, channels=9):
        super().__init__()
        self.channels = channels
        self.patch = nn.Linear(channels, 8)
        self.hidden = nn.Linear(512, 64)
        self.preview = nn.Linear(64, 6)

    def forward(self, x):
        patches = x.reshape(-1, 4, self.channels, 16).transpose(2, 3)
        return self.preview(self.hidden(self.patch(patches).tanh().flatten(1)).tanh())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, root = read(args.plan), Path(args.data)
    band = plan.get('feature_mode') == 'retinotopic_band'
    retinotopic = band or plan.get('feature_mode') == 'retinotopic_pair'
    expected_conditions = ['full_field_all_views', 'front_band_all_views', 'front_band_only'] if (
        band) else ['raw_control', 'processed_actual'] if retinotopic else (
            ['original_only', 'extended_train'])
    channels = 14 if retinotopic else 9
    if plan['conditions'] != expected_conditions or plan['epochs'] != 200:
        raise ValueError('Fixed paired conditions and last200 checkpoint required')
    manifest = read(root/'dataset.json')
    expected_schema = 'flybrain.retinotopic_preview.episodes.v2' if band else (
        'flybrain.retinotopic_preview.episodes.v1' if retinotopic else
        'flybrain.visual_preview.episodes.v1')
    if manifest['schema'] != expected_schema:
        raise ValueError('Explicit auxiliary-only visual dataset required')
    episodes = []
    for entry in manifest['episodes']:
        if entry['role'] not in ('train', 'development'):
            raise ValueError('No evaluation arrays may be opened')
        if (sha256(entry['features_file']) != entry['features_sha256']
                or sha256(entry['targets_file']) != entry['targets_sha256']):
            raise ValueError('Sealed original features and preview targets required')
        with load_array(entry['features_file'], allow_pickle=False) as f, load_array(
                entry['targets_file'], allow_pickle=False) as t:
            np.testing.assert_array_equal(f['observation_time'], t['observation_time'])
            y, valid = t['preview'].reshape(-1, 6), np.repeat(t['preview_valid'], 2, axis=1)
            features = {}
            if retinotopic:
                for key in expected_conditions:
                    if band:
                        ref = entry['feature_sources'][key]
                        if sha256(ref['file']) != ref['sha256']:
                            raise ValueError('Explicit condition feature reference changed')
                        with load_array(ref['file'], allow_pickle=False) as inputs:
                            np.testing.assert_array_equal(inputs['observation_time'],
                                                          f['observation_time'])
                            features[key] = inputs[ref['key']].copy()
                    else:
                        features[key] = f[key].copy()
            episodes.append(dict(entry, x=features['raw_control'] if retinotopic
                                 and not band else features['full_field_all_views'] if band
                                 else f['scene_features'].copy(), features=features,
                                 y=y.copy(), valid=valid.copy(),
                                 clock=f['observation_time'].copy()))
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU learning required')
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan,
        plan_sha256=sha256(args.plan), dataset_sha256=sha256(root/'dataset.json'),
        program_sha256=sha256(__file__), device=torch.cuda.get_device_name(0),
        development_used_only_for_terminal_reporting=True, actor_policy=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    records, reports = [], []
    for condition in plan['conditions']:
        if retinotopic:
            for episode in episodes:
                episode['x'] = episode['features'][condition]
        moments = training_moments(episodes,
            include_spatial=retinotopic or condition == 'extended_train',
            feature_floor=plan['feature_scale_floor'], target_floor=plan['target_scale_floor'],
            feature_count=4*channels*16)
        mean, scale = moments['feature_mean'], moments['feature_scale']
        ym, ys = moments['target_mean'], moments['target_scale']
        batch = torch.tensor(((moments['x']-mean)/scale).clip(-5, 5),
                             device='cuda', dtype=torch.float32)
        targets = torch.tensor((moments['y']-ym)/ys, device='cuda', dtype=torch.float32)
        weights = torch.tensor(moments['valid']/(6*moments['valid'].sum(0)),
                               device='cuda', dtype=torch.float32)
        torch.manual_seed(plan['seed'])
        model = PreviewBoundary(channels).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['weight_decay'])
        losses = []
        for epoch in range(plan['epochs']):
            optimizer.zero_grad(set_to_none=True)
            loss = (((model(batch)-targets)**2)*weights).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite preview loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite visual-boundary gradient')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if epoch % 25 == 0:
                write_json(run/'progress.json', dict(condition=condition, epoch=epoch+1,
                                                     loss=losses[-1]))
        model.eval()
        schema = 'flybrain.visual.retinotopic_preview_boundary.v2' if band else (
            'flybrain.visual.retinotopic_preview_boundary.v1' if retinotopic else
            'flybrain.visual.preview_boundary.v1')
        readout_contract = dict(channel_calibration_file=manifest.get('channel_calibration_file',
            str(root/'channel-calibration.npz')),
            channel_calibration_sha256=manifest['channel_calibration_sha256'],
            channel_order=manifest['processed_channels'] if band or condition == 'processed_actual'
                else manifest['raw_control_channels'],
            pooling_contract=manifest.get('pooling_contract'),
            scene_spec=plan['scene_spec'], condition=condition) if retinotopic else None
        torch.save(dict(schema=schema, channels=channels, model=model.state_dict(),
            feature_mean=mean, feature_scale=scale, target_mean=ym, target_scale=ys,
            condition=condition, plan=plan, readout_contract=readout_contract,
            actuator_policy=False), run/(condition+'.pt'))
        predictions = {}
        for episode in episodes:
            normalized = ((episode['x']-mean)/scale).clip(-5, 5)
            variants = dict(intact=normalized, vision_train_mean=np.zeros_like(normalized))
            interventions = [('motion_train_mean', slice(6, 14)),
                             ('intermediate_train_mean', slice(0, 6))] if retinotopic else [
                             ('motion_train_mean', slice(1, 9)),
                             ('intensity_train_mean', slice(0, 1))]
            for name, selected_channels in interventions:
                changed = normalized.reshape(-1, 4, channels, 16).copy()
                changed[:, :, selected_channels] = 0
                variants[name] = changed.reshape(-1, 4*channels*16)
            for intervention, x in variants.items():
                with torch.no_grad():
                    prediction = model(torch.tensor(x, device='cuda', dtype=torch.float32))
                    prediction = prediction.cpu().numpy()*ys+ym
                predictions[episode['id']+'__'+intervention] = prediction
                report = dict(condition=condition, episode=episode['id'], role=episode['role'],
                              group=episode['group'], intervention=intervention)
                for period in ('full', 'after3s'):
                    rows = []
                    for h in range(6):
                        keep = episode['valid'][:, h].copy()
                        if period == 'after3s':
                            keep &= episode['clock'] >= episode['clock'][0]+3
                        y, p = episode['y'][keep, h], prediction[keep, h]
                        rows.append(dict(count=int(keep.sum()),
                            rmse=float(np.sqrt(np.mean((p-y)**2))),
                            train_mean_rmse=float(np.sqrt(np.mean((ym[h]-y)**2))),
                            zero_rmse=float(np.sqrt(np.mean(y**2)))))
                    report[period] = rows
                reports.append(report)
        np.savez_compressed(run/(condition+'-predictions.npz'), **predictions)
        records.append(dict(condition=condition, fitting_histories=moments['fitting_histories'],
                            epochs=len(losses), losses=losses))
    write_json(run/'result.json', dict(status='completed_paired_spatial_preview_learning',
        records=records, reports=reports, target_order=[
            '5m_bearing_rad', '5m_chord_curvature_inv_m',
            '10m_bearing_rad', '10m_chord_curvature_inv_m', '20m_bearing_rad',
            '20m_chord_curvature_inv_m'], original_roles_and_clocks_preserved=True,
        target_clipping=False, actor_inputs=f'scene{4*channels*16} only',
        learned_boundary_is_engineering=True,
        author_visual_weights_or_motor_graph_changed=False, motor_transfer_performed=False,
        actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
