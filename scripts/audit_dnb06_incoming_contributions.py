"""Audit modeled presynaptic inputs to DNb06 on original visual histories."""

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
        types, ids = graph['types'].copy(), graph['ids'].copy()
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
    incoming = torch.nonzero(rows[1], as_tuple=False).flatten()
    if not 0 < len(incoming) <= 113:
        raise ValueError('Nonzero modeled DNb06_R input weights required')
    incoming_types = types[incoming.cpu().numpy()]
    incoming_ids = ids[incoming.cpu().numpy()]
    prior = read(Path(args.prior)/'result.json')
    if (prior['status'] != 'completed_dnb06_original_history_state_audit'
            or sha256(Path(args.prior)/'predictions.npz') != prior['predictions_sha256']):
        raise ValueError('Exact original history state proof required')
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = dataset['episodes']
    if sum(e['role'] == 'train' for e in entries) != 6 or sum(
            e['role'] == 'development' for e in entries) != 2:
        raise ValueError('Original six TRAIN and two known DEVELOPMENT histories required')
    conditions = ('intact', 'motion_mean')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'training_result_sha256': sha256(training/'result.json'),
        'candidate_checkpoint_sha256': trained['checkpoint_sha256'],
        'comparison_predictions_sha256': comparison['predictions_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'),
        'known_development_not_blind': True, 'conditions': conditions,
        'prior_predictions_sha256': prior['predictions_sha256'],
        'structural_incoming_edges': 113, 'nonzero_modeled_edges': len(incoming),
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
            drives = [intact_drive, mean_drive]
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays['observation_time'])]
            states = [model.initial_state(1) for _ in conditions]
            outputs, neurons, nets = ([[] for _ in conditions] for _ in range(3))
            held = None
            right_contributions = []
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        states = [model._advance_elapsed_tensor(held[k], states[k], dt)
                                  for k in range(len(conditions))]
                for k, state in enumerate(states):
                    outputs[k].append(model._read_state_tensor(state)[0][0])
                    neurons[k].append(state[0, positions])
                    nets[k].append((state@rows.T + drives[k][index:index+1, positions])[0])
                right_contributions.append(states[0][0, incoming]*rows[1, incoming])
                held = [drive[index:index+1] for drive in drives]
            predictions = [torch.stack(value).cpu().numpy() for value in outputs]
            traces = [torch.stack(value).cpu().numpy() for value in neurons]
            net_values = [torch.stack(value).cpu().numpy() for value in nets]
            prefix = entry['file'].removesuffix('.npz')
            contribution = torch.stack(right_contributions).cpu().numpy()
            saved[prefix+'_right_input_contributions'] = contribution
            with np.load(Path(args.prior)/'predictions.npz', allow_pickle=False) as old:
                state_error = float(np.max(np.abs(traces[0]-old[prefix+'_intact_state'])))
                net_error = float(np.max(np.abs(net_values[0]-old[prefix+'_intact_net'])))
            if max(state_error, net_error) > 1e-6:
                raise ValueError('Original state/net trajectory changed')
            parity = None
            if entry['role'] == 'development':
                with np.load(Path(args.comparison)/'predictions.npz', allow_pickle=False) as prior:
                    parity = [float(np.max(np.abs(predictions[0]-prior[prefix+'_intact']))),
                              float(np.max(np.abs(predictions[1]-prior[prefix+'_all_mean'])))]
                if max(parity) > 1e-6:
                    raise ValueError('Original/mean trajectories differ from prior replay')
            for k, condition in enumerate(conditions):
                saved[prefix+'_'+condition+'_state'] = traces[k]
                saved[prefix+'_'+condition+'_net'] = net_values[k]
                saved[prefix+'_'+condition+'_intent'] = predictions[k]
            report = {'file': entry['file'], 'role': entry['role'],
                'observations': len(schedule), 'prior_max_errors': parity, 'conditions': {}}
            for k, condition in enumerate(conditions):
                state, net = traces[k], net_values[k]
                report['conditions'][condition] = {
                    'state_mean': state.mean(axis=0).tolist(),
                    'state_max': state.max(axis=0).tolist(),
                    'net_min': net.min(axis=0).tolist(), 'net_max': net.max(axis=0).tolist(),
                    'net_positive_count': (net > 0).sum(axis=0).tolist(),
                    'state_above_1e_6_count': (state > 1e-6).sum(axis=0).tolist()}
            report['mean_abs_motion_state_change'] = np.abs(traces[0]-traces[1]).mean(0).tolist()
            reconstruction = contribution.sum(axis=1)+drives[0][:, positions[1]].cpu().numpy()
            reconstruction_error = float(np.max(np.abs(reconstruction-net_values[0][:, 1])))
            if reconstruction_error > 1e-6:
                raise ValueError('Right net-input reconstruction differs')
            report['net_reconstruction_max_error'] = reconstruction_error
            means = contribution.mean(axis=0)
            grouped = {}
            for label, value in zip(incoming_types, means, strict=True):
                key = str(label) or '<untyped>'
                grouped[key] = grouped.get(key, 0.)+float(value)
            report['right_positive_mean'] = float(np.maximum(contribution, 0).sum(1).mean())
            report['right_negative_mean'] = float(np.minimum(contribution, 0).sum(1).mean())
            report['right_direct_drive_mean'] = float(drives[0][:, positions[1]].mean())
            report['right_mean_top_inhibitory_types'] = sorted(
                grouped.items(), key=lambda x: x[1])[:8]
            report['right_mean_top_excitatory_types'] = sorted(
                grouped.items(), key=lambda x: x[1], reverse=True)[:8]
            report['prior_state_max_error'] = state_error
            report['prior_net_max_error'] = net_error
            reports.append(report)
            print(json.dumps(report), flush=True)
    saved['right_input_ids'] = incoming_ids
    saved['right_input_types'] = incoming_types
    np.savez_compressed(run/'predictions.npz', **saved)
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']):
        raise ValueError('Fixed checkpoint changed during replay')
    write_json(run/'result.json', {'status': 'completed_dnb06_original_history_input_audit',
        'reports': reports, 'predictions_sha256': sha256(run/'predictions.npz'),
        'known_development_not_independent': True, 'original_clocks_and_states_preserved': True,
        'net_is_current_decision_drive_not_exposure_synchronization': True,
        'feature_replacement_is_not_image_resimulation': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
