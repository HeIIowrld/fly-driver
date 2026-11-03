"""Probe causally delayed four-view motion with descending inputs active."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prior', required=True)
    args = parser.parse_args()
    training = Path(args.training)
    receipt = read(training/'result.json')
    manifest = read(training/'manifest.json')
    plan = manifest['config']['plan']
    if (receipt['status'] != 'completed_fixed_joint_command_training'
            or sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']):
        raise ValueError('Original fixed joint training checkpoint required')
    source = read(Path(plan['source'])/'result.json')
    if sha256(source['checkpoint']) != source['checkpoint_sha256']:
        raise ValueError('Original source checkpoint changed')
    if not torch.cuda.is_available():
        raise ValueError('GPU required for original cached graph')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'],
        torch.load(source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    checkpoint = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['spec'] != model.spec or checkpoint['schema'] != model.schema:
        raise ValueError('Original graph/spec mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    candidate = Path(args.candidate)
    trained = read(candidate/'result.json')
    if (trained['status'] != 'completed_one_pass_hsvs_boundary_train'
            or trained['source_checkpoint_sha256'] != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Exact completed HS/VS boundary candidate required')
    boundary = torch.load(candidate/'visual-boundary.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary['weight'].to('cuda'))
        model.backbone.vision.bias.copy_(boundary['bias'].to('cuda'))
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('Expected two known development histories')
    conditions = ('original', 'mean', 'delay_0_5', 'delay_2_0')
    prior = Path(args.prior)
    previous = read(prior/'result.json')
    previous_config = read(prior/'manifest.json')['config']
    if (previous['status'] != 'completed_four_view_motion_unclamped_factorial_replay'
            or sha256(prior/'predictions.npz') != previous['predictions_sha256']
            or previous_config['candidate_checkpoint_sha256'] != trained['checkpoint_sha256']
            or previous_config['dataset_sha256'] != sha256(root/'dataset.json')):
        raise ValueError('Exact prior clamped 16-condition replay required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'training_result_sha256': sha256(training/'result.json'),
        'candidate_checkpoint_sha256': trained['checkpoint_sha256'],
        'prior_predictions_sha256': previous['predictions_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'),
        'known_development_not_blind': True, 'conditions': conditions,
        'motion_replacement': 'all motion TRAIN mean or causal earlier motion; current brightness',
        'lags_seconds': [0.5, 2.0], 'invalid_prefix': 'motion TRAIN mean and mask',
        'direct_ports': 'original inputs; no motion clamp',
        'actor_deployed': False, 'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, saved = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root/entry['file']
            if sha256(path) != entry['sha256']:
                raise ValueError('Original episode changed')
            with load_array(path, allow_pickle=False) as a:
                arrays = {k: a[k].copy() for k in a.files}
            names = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')
            t = {k: torch.tensor(arrays[k], device='cuda') for k in names}
            if t['visual_valid'].any():
                raise ValueError('Original invalid functional visual6 mask must remain unchanged')
            time = arrays['observation_time']
            motion = torch.zeros((4, 9, 16), dtype=torch.bool, device='cuda')
            motion[:, 1:, :] = True
            motion = motion.flatten()
            scene = t['scene_features']
            mean = scene.clone()
            mean[:, motion] = model.backbone.scene_command.mean[motion]
            scenes, valid_lags, source_indices = [scene, mean], [], []
            for lag in (.5, 2.):
                indices = np.searchsorted(time, time - lag, side='right') - 1
                available = indices >= 0
                safe = np.clip(indices, 0, len(time)-1)
                available &= (time - time[safe] >= lag - 1e-9)
                available &= (time - time[safe] <= lag + .08)
                shifted = mean.clone()
                rows = torch.tensor(np.flatnonzero(available), device='cuda')
                sources = torch.tensor(safe[available], device='cuda')
                shifted[rows[:, None], torch.where(motion)[0]] = (
                    scene[sources[:, None], torch.where(motion)[0]])
                scenes.append(shifted)
                valid_lags.append(available)
                source_indices.append(np.where(available, safe, -1))
            drives = [model._drive_tensor(t['visual'], t['visual_valid'], t['body'],
                t['body_valid'], t['goals'], scene_value) for scene_value in scenes]
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays['observation_time'])]
            state = model.initial_state(len(conditions))
            outputs, held = [], None
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        state = model._advance_elapsed_tensor(held, state, dt)
                outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
                held = torch.cat([drive[index:index+1] for drive in drives])
            output = np.stack(outputs)
            predictions = [output[:, i, :] for i in range(len(conditions))]
            prefix = entry['file'].removesuffix('.npz')
            for condition, prediction in zip(conditions, predictions, strict=True):
                saved[prefix+'_'+condition] = prediction
            with load_array(prior/'predictions.npz', allow_pickle=False) as before:
                original_error = float(np.max(np.abs(predictions[0] - before[prefix+'_mask_00'])))
                mean_error = float(np.max(np.abs(predictions[1] - before[prefix+'_mask_15'])))
            if max(original_error, mean_error) > 1e-6:
                raise ValueError('Prior unclamped original and mean motion must reproduce')
            active = (arrays['target_valid'][:, 0] & valid_lags[0] & valid_lags[1]
                      & (time >= time[0] + 3.0))
            if not active.any():
                raise ValueError('Shared valid teacher and delayed-motion interval required')
            target = arrays['target'][:, 0]
            report = {'file': entry['file'], 'observations': len(schedule),
                      'shared_steer_label_count': int(active.sum()),
                      'prior_original_max_error': original_error,
                      'prior_mean_max_error': mean_error, 'conditions': {}}
            for condition, prediction in zip(conditions, predictions, strict=True):
                report['conditions'][condition] = {
                    'steer_rmse': float(np.sqrt(np.mean(
                        (prediction[active, 0] - target[active])**2))),
                    'steer_mean_abs_change_vs_original': float(np.mean(np.abs(
                        prediction[active, 0] - predictions[0][active, 0])))}
            report['zero_steer_rmse'] = float(np.sqrt(np.mean(target[active]**2)))
            saved[prefix+'_valid_0_5'] = valid_lags[0]
            saved[prefix+'_valid_2_0'] = valid_lags[1]
            saved[prefix+'_source_0_5'] = source_indices[0]
            saved[prefix+'_source_2_0'] = source_indices[1]
            saved[prefix+'_score_mask'] = active
            saved[prefix+'_time'] = time
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {
        'status': 'completed_four_view_motion_unclamped_delay_replay',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'direct_port_comparison_is_engineering_counterfactual': True,
        'feature_replacement_is_not_image_resimulation': True,
        'delayed_visual_current_body_is_synthetic_counterfactual': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
