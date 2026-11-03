"""Per-view HS/VS state clamps with fixed descending motion ports."""

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
        types, roles = graph['types'].copy(), graph['roles'].copy()
    masks = {'HS': np.char.startswith(types, 'HS'), 'VS': np.char.startswith(types, 'VS'),
             'DNg02': np.char.startswith(types, 'DNg02'), 'wing': roles == 'output'}
    if [int(m.sum()) for m in masks.values()] != [8, 34, 29, 67]:
        raise ValueError('Original graph group identity required')
    groups = {name: torch.tensor(np.flatnonzero(mask), device='cuda')
              for name, mask in masks.items()}
    hsvs = torch.cat([groups['HS'], groups['VS']])
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('Expected two known development histories')
    conditions = ('intact', 'front_mean', 'left_mean', 'right_mean', 'rear_mean',
                  'front_clamp', 'left_clamp', 'right_clamp', 'rear_clamp')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'training_result_sha256': sha256(training/'result.json'),
        'candidate_checkpoint_sha256': trained['checkpoint_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'),
        'known_development_not_blind': True, 'conditions': conditions,
        'motion_replacement': 'selected view T4/T5 pooled features at TRAIN mean',
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
            view_sets = ((), (0,), (1,), (2,), (3,), (0, 1, 2, 3))
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
            reference = drives[-1][:, direct_ports].clone()
            for drive in drives:
                drive[:, direct_ports] = reference
                if not torch.equal(drive[:, direct_ports], reference):
                    raise ValueError('Direct descending ports must match')
            drives = drives[:5]+[drives[0]]*4
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1) for _ in conditions]
            outputs = [[] for _ in conditions]
            diffs = {name: [[] for _ in range(4)] for name in groups}
            clamp_error = torch.zeros((), device='cuda')
            held = None
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        states = [model._advance_elapsed_tensor(held[k], states[k], dt)
                                  for k in range(len(conditions))]
                        for j in range(4):
                            states[j+5][:, hsvs] = states[j+1][:, hsvs]
                            clamp_error = torch.maximum(clamp_error, (states[j+5][:, hsvs]
                                                        - states[j+1][:, hsvs]).abs().max())
                for k, state in enumerate(states):
                    outputs[k].append(model._read_state_tensor(state)[0][0])
                for name, indices in groups.items():
                    for j in range(4):
                        diffs[name][j].append((states[0][:, indices]
                                               - states[j+1][:, indices])[0])
                held = [drive[index:index+1] for drive in drives]
            predictions = [torch.stack(value).cpu().numpy() for value in outputs]
            prefix = entry['file'].removesuffix('.npz')
            for condition, prediction in zip(conditions, predictions, strict=True):
                saved[prefix+'_'+condition] = prediction
            report = {'file': entry['file'], 'observations': len(schedule), 'views': {},
                      'clamp_max_error': float(clamp_error)}
            for j, name in enumerate(('front', 'left', 'right', 'rear')):
                natural = predictions[0]-predictions[j+1]
                clamp_effect = predictions[0]-predictions[j+5]
                groups_report = {}
                for group, per_view in diffs.items():
                    values = torch.stack(per_view[j]).cpu().numpy()
                    saved[prefix+'_'+name+'_'+group] = values
                    groups_report[group] = float(np.abs(values).mean())
                report['views'][name] = {
                    'natural_readout_mean_abs': np.abs(natural).mean(axis=0).tolist(),
                    'hsvs_clamp_readout_mean_abs': np.abs(clamp_effect).mean(axis=0).tolist(),
                    'mean_abs_group_state_difference': groups_report}
            report['direct_ports_identical'] = True
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {'status': 'completed_four_view_hsvs_state_clamp',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'state_clamp_is_engineering_counterfactual_not_physiological_lesion': True,
        'feature_replacement_is_not_image_resimulation': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
