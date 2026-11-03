"""Targeted modeled edges to DNb06_R on original visual histories."""

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
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--prior', required=True)
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
    comparison = read(Path(args.comparison)/'result.json')
    if (comparison['status'] != 'completed_four_view_isolated_motion_interventions'
            or sha256(Path(args.comparison)/'predictions.npz') != comparison['predictions_sha256']
            or sha256(spec['graph_file']) != spec['graph_sha256']):
        raise ValueError('Pinned graph and previous prediction evidence required')
    positions = torch.tensor(np.flatnonzero(types == 'DNb06'), device='cuda')
    with load_array(spec['graph_file'], allow_pickle=False) as graph:
        if list(graph['ids'][positions.cpu().numpy()]) != ['10888', '11067']:
            raise ValueError('Exact DNb06 left/right identities required')
    matrix = model.backbone.circuit.matrix.coalesce()
    index, values = matrix.indices(), matrix.values()
    rows = torch.zeros((2, model.backbone.neurons), device='cuda')
    for j, position in enumerate(positions):
        selected = index[0] == position
        rows[j, index[1, selected]] = values[selected]
    prior = read(Path(args.prior)/'result.json')
    if (prior['status'] != 'completed_dnb06_original_history_state_audit'
            or sha256(Path(args.prior)/'predictions.npz') != prior['predictions_sha256']):
        raise ValueError('Exact prior fixed-state evidence required')
    selected_types = {'PS213': ('PS213',), 'IN06A124': ('IN06A124',),
        'both_negative': ('PS213', 'IN06A124'), 'DNge043': ('DNge043',), 'VST2': ('VST2',)}
    edge_rows = {}
    for group, labels in selected_types.items():
        mask = torch.tensor(np.isin(types, labels), device='cuda')
        edge_rows[group] = rows[1]*mask
        if not torch.any(edge_rows[group] != 0):
            raise ValueError('Each selected source must have modeled edges to DNb06_R')
    if (not torch.all(edge_rows['PS213'][edge_rows['PS213'] != 0] < 0)
            or not torch.all(edge_rows['IN06A124'][edge_rows['IN06A124'] != 0] < 0)
            or not torch.all(edge_rows['DNge043'][edge_rows['DNge043'] != 0] > 0)
            or not torch.all(edge_rows['VST2'][edge_rows['VST2'] != 0] > 0)):
        raise ValueError('Selected model signs differ from previous decomposition')
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = dataset['episodes']
    if sum(e['role'] == 'train' for e in entries) != 6 or sum(
            e['role'] == 'development' for e in entries) != 2:
        raise ValueError('Original six TRAIN and two known DEVELOPMENT histories required')
    conditions = ('intact', *edge_rows.keys())
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'training_result_sha256': sha256(training/'result.json'),
        'candidate_checkpoint_sha256': trained['checkpoint_sha256'],
        'comparison_predictions_sha256': comparison['predictions_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'),
        'known_development_not_blind': True, 'conditions': conditions,
        'prior_predictions_sha256': prior['predictions_sha256'],
        'intervention': 'subtract selected current source-to-DNb06_R edge terms per substep',
        'net_input': 'current state and current decision drive, not simulated exposure time',
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
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1) for _ in conditions]
            outputs, neurons = ([[] for _ in conditions] for _ in range(2))
            held = None
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        next_states = []
                        for k, state in enumerate(states):
                            drive = held
                            if k:
                                drive = held.clone()
                                removed = state@edge_rows[conditions[k]][:, None]
                                drive[:, positions[1]] -= removed[:, 0]
                            next_states.append(model._advance_elapsed_tensor(drive, state, dt))
                        states = next_states
                for k, state in enumerate(states):
                    outputs[k].append(model._read_state_tensor(state)[0][0])
                    neurons[k].append(state[0, positions])
                held = intact_drive[index:index+1]
            predictions = [torch.stack(value).cpu().numpy() for value in outputs]
            traces = [torch.stack(value).cpu().numpy() for value in neurons]
            prefix = entry['file'].removesuffix('.npz')
            with np.load(Path(args.prior)/'predictions.npz', allow_pickle=False) as old:
                parity_output = float(np.max(np.abs(predictions[0]-old[prefix+'_intact_intent'])))
                parity_state = float(np.max(np.abs(traces[0]-old[prefix+'_intact_state'])))
            if max(parity_output, parity_state) > 1e-6:
                raise ValueError('Unchanged model must match original replay')
            report = {'file': entry['file'], 'role': entry['role'],
                'observations': len(schedule), 'baseline_prior_max_error':
                max(parity_output, parity_state), 'conditions': {}}
            for k, condition in enumerate(conditions):
                saved[prefix+'_'+condition+'_state'] = traces[k]
                saved[prefix+'_'+condition+'_intent'] = predictions[k]
                valid = arrays['target_valid'][:, 0]
                rmse = float(np.sqrt(np.mean((predictions[k][valid, 0]
                                              - arrays['target'][valid, 0])**2)))
                report['conditions'][condition] = {
                    'right_mean_state': float(traces[k][:, 1].mean()),
                    'right_above_1e_6_count': int((traces[k][:, 1] > 1e-6).sum()),
                    'mean_abs_steering_difference': float(np.abs(
                        predictions[k][:, 0]-predictions[0][:, 0]).mean()),
                    'steering_teacher_rmse': rmse}
            reports.append(report)
            print(json.dumps(report), flush=True)
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {'status': 'completed_dnb06_incoming_edge_interventions',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'targeted_model_edge_removal_not_biological_lesion': True,
        'feature_replacement_is_not_image_resimulation': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
