"""Clamp selected measured-graph descending candidates during four-view motion replay."""

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
    parser.add_argument('--route-audit', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
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
    spec = read(plan['source_spec'])
    with load_array(spec['graph_file'], allow_pickle=False) as graph:
        types = graph['types'].copy()
    route = read(Path(args.route_audit)/'result.json')
    comparison = read(Path(args.comparison)/'result.json')
    if (route['status'] != 'completed_visual_flight_structural_path_audit'
            or comparison['status'] != 'completed_four_view_isolated_motion_interventions'
            or sha256(Path(args.comparison)/'predictions.npz') != comparison['predictions_sha256']
            or sha256(spec['graph_file']) != spec['graph_sha256']):
        raise ValueError('Pinned graph route and fixed previous prediction evidence required')
    names_of_groups = ('DNbe001', 'DNp18', 'DNge107', 'DNb06')
    groups = {name: torch.tensor(np.flatnonzero(types == name), device='cuda')
              for name in names_of_groups}
    if [len(groups[name]) for name in names_of_groups] != [2, 2, 2, 2]:
        raise ValueError('Expected graph candidate identities required')
    groups['all_four'] = torch.cat(list(groups.values()))
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('Expected two known development histories')
    conditions = ('intact', 'motion_mean', *groups.keys())
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'training_result_sha256': sha256(training/'result.json'),
        'candidate_checkpoint_sha256': trained['checkpoint_sha256'],
        'route_audit_sha256': sha256(Path(args.route_audit)/'result.json'),
        'comparison_predictions_sha256': comparison['predictions_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'),
        'known_development_not_blind': True, 'conditions': conditions,
        'motion_replacement': 'all four T4/T5 pooled features at TRAIN mean',
        'direct_ports': 'all motion features TRAIN mean for every condition',
        'actor_deployed': False, 'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, saved = [], {}
    direct_ports = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
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
            mean_scene = t['scene_features'].clone()
            select = torch.zeros(576, dtype=torch.bool, device='cuda').reshape(4, 9, 16)
            select[:, 1:] = True
            select = select.flatten()
            mean_scene[:, select] = model.backbone.scene_command.mean[select]
            intact_drive = model._drive_tensor(*(t[k] for k in names))
            mean_drive = model._drive_tensor(*(t[k] for k in names[:5]), mean_scene)
            intact_drive[:, direct_ports] = mean_drive[:, direct_ports]
            if not torch.equal(intact_drive[:, direct_ports], mean_drive[:, direct_ports]):
                raise ValueError('Exact direct descending drive equality required')
            drives = [intact_drive, mean_drive]+[intact_drive]*len(groups)
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1) for _ in conditions]
            outputs = [[] for _ in conditions]
            clamp_error = torch.zeros((), device='cuda')
            held = None
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        states = [model._advance_elapsed_tensor(held[k], states[k], dt)
                                  for k in range(len(conditions))]
                        for j, indices in enumerate(groups.values()):
                            states[j+2][:, indices] = states[1][:, indices]
                            clamp_error = torch.maximum(clamp_error, (states[j+2][:, indices]
                                                        - states[1][:, indices]).abs().max())
                for k, state in enumerate(states):
                    outputs[k].append(model._read_state_tensor(state)[0][0])
                held = [drive[index:index+1] for drive in drives]
            predictions = [torch.stack(value).cpu().numpy() for value in outputs]
            prefix = entry['file'].removesuffix('.npz')
            with np.load(Path(args.comparison)/'predictions.npz', allow_pickle=False) as prior:
                parity_intact = float(np.max(np.abs(predictions[0]-prior[prefix+'_intact'])))
                parity_mean = float(np.max(np.abs(predictions[1]-prior[prefix+'_all_mean'])))
            if max(parity_intact, parity_mean) > 1e-6:
                raise ValueError('Original/mean trajectories differ from prior replay')
            for condition, prediction in zip(conditions, predictions, strict=True):
                saved[prefix+'_'+condition] = prediction
            natural = predictions[0]-predictions[1]
            report = {'file': entry['file'], 'observations': len(schedule), 'groups': {},
                'intact_prior_max_error': parity_intact, 'mean_prior_max_error': parity_mean,
                'clamp_max_error': float(clamp_error),
                'motion_mean_readout_change': np.abs(natural).mean(axis=0).tolist()}
            for j, name in enumerate(groups):
                change = predictions[0]-predictions[j+2]
                remaining = predictions[j+2]-predictions[1]
                report['groups'][name] = {'neurons': len(groups[name]),
                    'clamp_readout_change': np.abs(change).mean(axis=0).tolist(),
                    'clamp_minus_mean_readout': np.abs(remaining).mean(axis=0).tolist()}
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {'status': 'completed_visual_descending_state_clamp',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'state_clamp_is_engineering_counterfactual_not_physiological_lesion': True,
        'feature_replacement_is_not_image_resimulation': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
