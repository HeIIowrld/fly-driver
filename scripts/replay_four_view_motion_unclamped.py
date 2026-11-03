"""Compare four-view motion subsets with naturally varying descending inputs."""

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
    conditions = tuple(f'mask_{i:02d}' for i in range(16))
    prior = Path(args.prior)
    previous = read(prior/'result.json')
    previous_config = read(prior/'manifest.json')['config']
    if (previous['status'] != 'completed_four_view_motion_factorial_replay'
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
        'motion_replacement': 'selected view T4/T5 pooled features at TRAIN mean',
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
            view_sets = [tuple(v for v in range(4) if mask & (1 << v))
                         for mask in range(16)]
            drives = []
            for selected_views in view_sets:
                scene = t['scene_features'].clone()
                select = torch.zeros(576, dtype=torch.bool, device='cuda').reshape(4, 9, 16)
                for view in selected_views:
                    select[view, 1:] = True
                select = select.flatten()
                scene[:, select] = model.backbone.scene_command.mean[select]
                drives.append(model._drive_tensor(t['visual'], t['visual_valid'], t['body'],
                    t['body_valid'], t['goals'], scene))
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
                clamped_full = before[prefix+'_mask_00'].copy()
                clamped_mean = before[prefix+'_mask_15'].copy()
            mismatch = float(np.max(np.abs(predictions[15] - clamped_mean)))
            if mismatch > 1e-6:
                raise ValueError('All-motion-mean baseline must match clamped replay')
            contrast_unclamped = predictions[0] - predictions[15]
            contrast_clamped = clamped_full - clamped_mean
            difference = contrast_unclamped - contrast_clamped
            pairs = {}
            view_names = ('front', 'left', 'right', 'rear')
            for i in range(4):
                for j in range(i+1, 4):
                    pair = predictions[0] - predictions[1 << i] - predictions[1 << j] + (
                        predictions[(1 << i) | (1 << j)])
                    pairs[view_names[i]+'_'+view_names[j]] = {
                        'interaction_mean_abs': np.mean(np.abs(pair), axis=0).tolist(),
                        'interaction_max_abs': np.max(np.abs(pair), axis=0).tolist()}
            mean = predictions[15]
            additive = sum((predictions[15 ^ (1 << v)] - mean for v in range(4)))
            residual = predictions[0] - mean - additive
            report = {'file': entry['file'], 'observations': len(schedule),
                      'prior_all_mean_max_error': mismatch, 'pairs': pairs,
                      'unclamped_minus_clamped_motion_contrast_mean_abs': np.mean(
                          np.abs(difference), axis=0).tolist(),
                      'full_vs_sum_isolated_interaction_mean_abs': np.mean(
                          np.abs(residual), axis=0).tolist(),
                      'full_vs_mean_motion_mean_abs': np.mean(
                          np.abs(predictions[0] - mean), axis=0).tolist(),
                      'direct_ports_clamped': False}
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {
        'status': 'completed_four_view_motion_unclamped_factorial_replay',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'direct_port_comparison_is_engineering_counterfactual': True,
        'feature_replacement_is_not_image_resimulation': True,
        'interaction_is_not_physiological_synergy': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
