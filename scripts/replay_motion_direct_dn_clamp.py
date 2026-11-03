"""Offline direct descending-input clamp under motion-feature intervention."""

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
    spec = read(plan['source_spec'])
    with load_array(spec['graph_file'], allow_pickle=False) as graph:
        types, roles = graph['types'].copy(), graph['roles'].copy()
    masks = {'HS': np.char.startswith(types, 'HS'), 'VS': np.char.startswith(types, 'VS'),
        'DNg02': np.char.startswith(types, 'DNg02'), 'wing_motor_candidates': roles == 'output'}
    if [int(m.sum()) for m in masks.values()] != [8, 34, 29, 67]:
        raise ValueError('Expected original measured-graph candidate populations')
    groups = {name: torch.tensor(np.flatnonzero(mask), device='cuda')
              for name, mask in masks.items()}
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('Expected original two known development histories')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'training_result_sha256': sha256(training/'result.json'),
        'checkpoint_sha256': receipt['checkpoint_sha256'], 'known_development_not_blind': True,
        'motion_intervention': 'all four 8 T4/T5 pooled channels at TRAIN feature mean',
        'direct_dn_clamp': 'descending steering/goal drive set to motion-mean condition',
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
            edited = t['scene_features'].clone()
            select = torch.zeros(576, dtype=torch.bool, device='cuda').reshape(4, 9, 16)
            select[:, 1:] = True
            select = select.flatten()
            edited[:, select] = model.backbone.scene_command.mean[select]
            drives = [model._drive_tensor(t['visual'], t['visual_valid'], t['body'],
                t['body_valid'], t['goals'], scene) for scene in (t['scene_features'], edited)]
            direct_ports = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
            blocked_direct = drives[0].clone()
            blocked_direct[:, direct_ports] = drives[1][:, direct_ports]
            reference_port_error = (blocked_direct[:, direct_ports]
                                    - drives[1][:, direct_ports]).abs().max()
            protected_vision_error = (blocked_direct[:, model.backbone.vision_ports]
                                      - drives[0][:, model.backbone.vision_ports]).abs().max()
            if float(reference_port_error) != 0 or float(protected_vision_error) != 0:
                raise ValueError('Exact descending replacement and intact vision required')
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1), model.initial_state(1), model.initial_state(1)]
            held = None
            collected = {name: [] for name in groups}
            clamp_collected = {name: [] for name in groups}
            outputs, clamp_outputs = [], []
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                            states = [model._advance_elapsed_tensor(held[k], states[k], dt)
                                      for k in range(3)]
                for name, positions in groups.items():
                    collected[name].append((states[0][:, positions]-states[1][:, positions])[0])
                    clamp_collected[name].append(
                        (states[0][:, positions]-states[2][:, positions])[0])
                outputs.append((model._read_state_tensor(states[0])[0]
                                - model._read_state_tensor(states[1])[0])[0])
                clamp_outputs.append((model._read_state_tensor(states[0])[0]
                                      - model._read_state_tensor(states[2])[0])[0])
                held = [drives[0][index:index+1], drives[1][index:index+1],
                        blocked_direct[index:index+1]]
            report = {'file': entry['file'], 'observations': len(schedule), 'groups': {}}
            for name, values in collected.items():
                delta = torch.stack(values).cpu().numpy()
                saved[entry['file'].removesuffix('.npz')+'_'+name] = delta
                report['groups'][name] = {'neurons': int(delta.shape[1]),
                    'mean_absolute_state_difference': float(np.abs(delta).mean()),
                    'maximum_absolute_state_difference': float(np.abs(delta).max())}
                clamped = torch.stack(clamp_collected[name]).cpu().numpy()
                saved[entry['file'].removesuffix('.npz')+'_clamp_'+name] = clamped
                report['groups'][name]['clamp_mean_absolute_state_difference'] = float(
                    np.abs(clamped).mean())
            output = torch.stack(outputs).cpu().numpy()
            saved[entry['file'].removesuffix('.npz')+'_intent'] = output
            report['intent_mean_abs_difference'] = np.abs(output).mean(axis=0).tolist()
            clamp_output = torch.stack(clamp_outputs).cpu().numpy()
            saved[entry['file'].removesuffix('.npz')+'_clamp_intent'] = clamp_output
            report['clamp_intent_mean_abs_difference'] = np.abs(clamp_output).mean(axis=0).tolist()
            report['direct_ports_reference_max_error'] = float(reference_port_error)
            report['vision_ports_intact_max_error'] = float(protected_vision_error)
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'group-differences.npz', **saved)
    if sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']:
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {'status': 'completed_direct_descending_clamp_replay',
        'reports': reports, 'group_difference_sha256': sha256(run/'group-differences.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'direct_port_clamp_is_engineering_counterfactual_not_physiological_lesion': True,
        'motion_feature_is_not_image_resimulation': True,
        'group_annotations_are_not_functional_proof': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
