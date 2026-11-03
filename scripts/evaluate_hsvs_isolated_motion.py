"""Compare fixed HS/VS candidates with direct DN motion features at TRAIN mean."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.evaluate_embodied_flight_navigation import metrics
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training, candidate = Path(args.training), Path(args.candidate)
    original = read(training/'result.json')
    result = read(candidate/'result.json')
    plan = read(training/'manifest.json')['config']['plan']
    if (original['status'] != 'completed_fixed_joint_command_training'
            or result['status'] != 'completed_one_pass_hsvs_boundary_train'
            or sha256(original['checkpoint']) != original['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != result['checkpoint_sha256']
            or result['source_checkpoint_sha256'] != original['checkpoint_sha256']
            or not result['direct_dn_parameters_and_graph_unchanged']
            or not result['direct_dn_motion_features_held_at_train_mean']):
        raise ValueError('Exact original and completed fixed candidate required')
    source = read(Path(plan['source'])/'result.json')
    if sha256(source['checkpoint']) != source['checkpoint_sha256'] or not torch.cuda.is_available():
        raise ValueError('Original source checkpoint and GPU required')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'],
        torch.load(source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    checkpoint = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    boundary = torch.load(candidate/'visual-boundary.pt', map_location='cpu', weights_only=True)
    if boundary['source_checkpoint_sha256'] != original['checkpoint_sha256']:
        raise ValueError('Candidate was fitted from different checkpoint')
    old_weight, old_bias = model.backbone.vision.weight.detach().clone(), (
        model.backbone.vision.bias.detach().clone())
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = dataset['episodes']
    if sum(e['role'] == 'train' for e in entries) != 6 or sum(
            e['role'] == 'development' for e in entries) != 2:
        raise ValueError('Original six TRAIN and two known DEVELOPMENT roles required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'original_checkpoint_sha256': original['checkpoint_sha256'],
        'candidate_checkpoint_sha256': result['checkpoint_sha256'],
        'direct_motion_path': 'steering/goal ports use motion-mean counterfactual drive',
        'development_known_not_blind': True, 'optimizer_updates': 0,
        'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, saved = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root/entry['file']
            if sha256(path) != entry['sha256']:
                raise ValueError('Original history changed')
            with load_array(path, allow_pickle=False) as a:
                arrays = {k: a[k].copy() for k in a.files}
            names = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')
            data = {k: torch.tensor(arrays[k], device='cuda') for k in names}
            if data['visual_valid'].any():
                raise ValueError('Original functional visual6 mask must remain invalid')
            with torch.no_grad():
                model.backbone.vision.weight.copy_(old_weight)
                model.backbone.vision.bias.copy_(old_bias)
                old_drive = model._drive_tensor(*(data[k] for k in names))
                mean_scene = data['scene_features'].clone()
                motion = torch.zeros(576, dtype=torch.bool, device='cuda').reshape(4, 9, 16)
                motion[:, 1:] = True
                motion = motion.flatten()
                mean_scene[:, motion] = model.backbone.scene_command.mean[motion]
                mean_drive = model._drive_tensor(*(data[k] for k in names[:5]), mean_scene)
                model.backbone.vision.weight.copy_(boundary['weight'].to('cuda'))
                model.backbone.vision.bias.copy_(boundary['bias'].to('cuda'))
                new_drive = model._drive_tensor(*(data[k] for k in names))
            direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
            old_drive[:, direct] = mean_drive[:, direct]
            new_drive[:, direct] = mean_drive[:, direct]
            direct_difference = float((old_drive[:, direct]-new_drive[:, direct]).abs().max())
            if direct_difference != 0:
                raise ValueError('Direct descending drive changed')
            schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1), model.initial_state(1)]
            previous, predictions = None, [[], []]
            for i, steps in enumerate(schedule):
                if i:
                    for dt in steps:
                        states = [model._advance_elapsed_tensor(previous[k], states[k], dt)
                                  for k in range(2)]
                for k in range(2):
                    predictions[k].append(model._read_state_tensor(states[k])[0][0])
                previous = [old_drive[i:i+1], new_drive[i:i+1]]
            old, new = [torch.stack(x).cpu().numpy() for x in predictions]
            if not np.isfinite(old).all() or not np.isfinite(new).all():
                raise ValueError('Finite original and candidate outputs required')
            prefix = entry['file'].removesuffix('.npz')
            saved[prefix+'_original'], saved[prefix+'_candidate'] = old, new
            reports.append({'file': entry['file'], 'role': entry['role'],
                'original': metrics(old, arrays['target'], arrays['target_valid']),
                'candidate': metrics(new, arrays['target'], arrays['target_valid']),
                'mean_abs_output_change': np.abs(new-old).mean(axis=0).tolist(),
                'maximum_direct_port_drive_difference': direct_difference})
            print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(original['checkpoint']) != original['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != result['checkpoint_sha256']):
        raise ValueError('Original or candidate checkpoint changed during evaluation')
    write_json(run/'result.json', {'status': 'completed_hsvs_isolated_motion_comparison',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'same_direct_dn_inputs_verified': True,
        'direct_dn_motion_features_replaced_by_training_mean': True,
        'original_history_and_states_preserved': True, 'positive_brake_labels_absent': True,
        'optimizer_updates': 0, 'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
