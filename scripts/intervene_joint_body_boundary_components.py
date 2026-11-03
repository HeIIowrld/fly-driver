"""Ablate trained body-boundary components with fixed original visual flight graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.graph import sha256
from scripts.evaluate_embodied_flight_navigation import metrics
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--visual-candidate', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training, candidate = Path(args.training), Path(args.candidate)
    original = read(training/'result.json')
    result = read(candidate/'result.json')
    visual_candidate = Path(args.visual_candidate)
    visual_result = read(visual_candidate/'result.json')
    plan = read(training/'manifest.json')['config']['plan']
    if (original['status'] != 'completed_fixed_joint_command_training'
            or result['status'] != 'completed_one_pass_joint_body_train'
            or sha256(original['checkpoint']) != original['checkpoint_sha256']
            or sha256(candidate/'body-boundary.pt') != result['checkpoint_sha256']
            or sha256(visual_candidate/'visual-boundary.pt') != result['visual_candidate_sha256']
            or visual_result['checkpoint_sha256'] != result['visual_candidate_sha256']
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
    boundary = torch.load(visual_candidate/'visual-boundary.pt', map_location='cpu',
                          weights_only=True)
    body = torch.load(candidate/'body-boundary.pt', map_location='cpu', weights_only=True)
    if (boundary['source_checkpoint_sha256'] != original['checkpoint_sha256']
            or body['visual_candidate_sha256'] != result['visual_candidate_sha256']):
        raise ValueError('Visual and body candidate provenance differs')
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary['weight'].to('cuda'))
        model.backbone.vision.bias.copy_(boundary['bias'].to('cuda'))
    joint = ExperimentalVisualBodyAirflow(model).to('cuda').eval()
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('Original two known DEVELOPMENT histories required')
    comparison = Path(args.comparison)
    compared = read(comparison/'result.json')
    if (compared['status'] != 'completed_joint_body_visual_comparison'
            or sha256(comparison/'predictions.npz') != compared['predictions_sha256']):
        raise ValueError('Exact fixed comparison predictions required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'original_checkpoint_sha256': original['checkpoint_sha256'],
        'candidate_checkpoint_sha256': result['checkpoint_sha256'],
        'comparison_predictions_sha256': compared['predictions_sha256'],
        'intervention': 'zero selected adapter weight columns, original body remains unchanged',
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
            count = len(data['body'])
            views = torch.ones((count, 4), dtype=torch.bool, device='cuda')
            airflow = torch.zeros((count, 3), device='cuda')
            airflow_valid = torch.zeros((count, 3), dtype=torch.bool, device='cuda')
            mean_scene = data['scene_features'].clone()
            motion = torch.zeros(576, dtype=torch.bool, device='cuda').reshape(4, 9, 16)
            motion[:, 1:] = True
            motion = motion.flatten()
            mean_scene[:, motion] = model.backbone.scene_command.mean[motion]
            mean_drive = model._drive_tensor(*(data[k] for k in names[:5]), mean_scene)
            direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
            conditions = {'full': [], 'neutral': list(range(28)), 'tilt': [0, 1],
                          'load': list(range(2, 6)), 'temperature': list(range(6, 10)),
                          'wheel_rate': list(range(10, 14)), 'availability': list(range(14, 28))}
            drives = []
            original_weight = body['weight'].to('cuda')
            for columns in conditions.values():
                joint.body_feedback.projection.weight.copy_(original_weight)
                if columns:
                    joint.body_feedback.projection.weight[:, columns] = 0
                drive = joint.drive(*(data[k] for k in names), views, airflow, airflow_valid)
                drive[:, direct] = mean_drive[:, direct]
                drives.append(drive)
            schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1) for _ in conditions]
            previous, predictions = None, [[] for _ in conditions]
            for i, steps in enumerate(schedule):
                if i:
                    for dt in steps:
                        states = [model._advance_elapsed_tensor(previous[k], states[k], dt)
                                  for k in range(len(conditions))]
                for k, state in enumerate(states):
                    predictions[k].append(model._read_state_tensor(state)[0][0])
                previous = [drive[i:i+1] for drive in drives]
            predictions = [torch.stack(x).cpu().numpy() for x in predictions]
            prefix = entry['file'].removesuffix('.npz')
            with np.load(comparison/'predictions.npz', allow_pickle=False) as old:
                full_error = float(np.max(np.abs(predictions[0]-old[prefix+'_candidate'])))
                neutral_error = float(np.max(np.abs(predictions[1]-old[prefix+'_original'])))
            if max(full_error, neutral_error) > 1e-6:
                raise ValueError('Full/neutral conditions differ from fixed comparison')
            report = {'file': entry['file'], 'role': entry['role'], 'conditions': {},
                      'full_previous_max_error': full_error,
                      'neutral_previous_max_error': neutral_error}
            for name, prediction in zip(conditions, predictions, strict=True):
                saved[prefix+'_'+name] = prediction
                report['conditions'][name] = {
                    'metrics': metrics(prediction, arrays['target'], arrays['target_valid']),
                    'mean_abs_difference_from_full': np.abs(prediction-predictions[0])
                        .mean(axis=0).tolist()}
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(original['checkpoint']) != original['checkpoint_sha256']
            or sha256(candidate/'body-boundary.pt') != result['checkpoint_sha256']
            or sha256(visual_candidate/'visual-boundary.pt') != result['visual_candidate_sha256']
            or visual_result['checkpoint_sha256'] != result['visual_candidate_sha256']):
        raise ValueError('Original or candidate checkpoint changed during evaluation')
    write_json(run/'result.json', {'status': 'completed_joint_body_component_interventions',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'same_direct_dn_inputs_verified': True,
        'only_adapter_weight_columns_ablated': True,
        'direct_dn_motion_features_replaced_by_training_mean': True,
        'original_history_and_states_preserved': True, 'positive_brake_labels_absent': True,
        'optimizer_updates': 0, 'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
