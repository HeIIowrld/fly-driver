"""Synthetic direction-channel feature pulses through the fixed measured flight graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training, candidate = Path(args.training), Path(args.candidate)
    receipt, plan = read(training/'result.json'), read(training/'manifest.json')['config']['plan']
    selected = read(candidate/'result.json')
    if (receipt['status'] != 'completed_fixed_joint_command_training'
            or selected['status'] != 'completed_one_pass_hsvs_boundary_train'
            or sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != selected['checkpoint_sha256']
            or selected['source_checkpoint_sha256'] != receipt['checkpoint_sha256']):
        raise ValueError('Exact frozen graph and HS/VS candidate required')
    source = read(Path(plan['source'])/'result.json')
    if sha256(source['checkpoint']) != source['checkpoint_sha256'] or not torch.cuda.is_available():
        raise ValueError('Original source graph and GPU required')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'],
        torch.load(source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    payload = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    boundary = torch.load(candidate/'visual-boundary.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary['weight'].to('cuda'))
        model.backbone.vision.bias.copy_(boundary['bias'].to('cuda'))
    spec = read(plan['source_spec'])
    if sha256(spec['graph_file']) != spec['graph_sha256']:
        raise ValueError('Original graph changed')
    with load_array(spec['graph_file'], allow_pickle=False) as graph:
        types = graph['types'].copy()
    positions = torch.tensor(np.flatnonzero(types == 'DNb06'), device='cuda')
    if len(positions) != 2:
        raise ValueError('Exact original DNb06 pair required')
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entry = next(e for e in dataset['episodes'] if e['role'] == 'train')
    episode = root/entry['file']
    if sha256(episode) != entry['sha256']:
        raise ValueError('Original context episode changed')
    with load_array(episode, allow_pickle=False) as data:
        values = {k: torch.tensor(data[k][0:1], device='cuda') for k in
                  ('visual', 'visual_valid', 'body', 'body_valid', 'goals')}
    conditions = [{'view': view, 'channel': channel, 'sign': sign}
                  for view in range(4) for channel in range(8) for sign in (-1, 1)]
    count = len(conditions)+1
    scene = model.backbone.scene_command.mean[None].repeat(count, 1)
    shaped = scene.reshape(count, 4, 9, 16)
    scale = model.backbone.scene_command.scale.reshape(4, 9, 16)
    for index, condition in enumerate(conditions, 1):
        view, channel, sign = condition['view'], condition['channel']+1, condition['sign']
        shaped[index, view, channel, :] += sign*scale[view, channel, :]
    repeated = {key: value.repeat(count, 1) for key, value in values.items()}
    with torch.no_grad():
        active_drive = model._drive_tensor(*(repeated[k] for k in values), scene)
        baseline_scene = model.backbone.scene_command.mean[None].repeat(count, 1)
        baseline_drive = model._drive_tensor(*(repeated[k] for k in values), baseline_scene)
        direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
        active_drive[:, direct] = baseline_drive[:, direct]
        if not torch.equal(active_drive[:, direct], baseline_drive[:, direct]):
            raise ValueError('Direct motion bypass must be held to baseline')
        run = begin_run(args.output, {'program_sha256': sha256(__file__),
            'checkpoint_sha256': receipt['checkpoint_sha256'],
            'visual_candidate_sha256': selected['checkpoint_sha256'],
            'graph_sha256': spec['graph_sha256'], 'episode_sha256': sha256(episode),
            'stimulus': 'synthetic pooled features mean +/- 1 TRAIN std, 16 spatial bins',
            'schedule_seconds': [0.5, 0.25, 0.75], 'dt_seconds': .01,
            'direct_motion_ports_clamped': True, 'optimizer_updates': 0})
        (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
        state = model.initial_state(count)
        traces, output, net_snapshots = [], [], []
        for step in range(150):
            drive = active_drive if 50 <= step < 75 else baseline_drive
            state = model._advance_elapsed_tensor(drive, state, .01)
            traces.append(state[:, positions].detach().cpu().numpy())
            output.append(model._read_state_tensor(state)[0].detach().cpu().numpy())
            if step in (49, 74, 124):
                matrix = model.backbone.circuit._cached_matrix
                net = torch.sparse.mm(matrix, state.T).T + drive
                net_snapshots.append(net[:, positions].detach().cpu().numpy())
            if (step+1) % 50 == 0:
                print(json.dumps({'completed_ticks': step+1}), flush=True)
    traces, output = np.stack(traces), np.stack(output)
    if not np.isfinite(traces).all() or not np.isfinite(output).all():
        raise ValueError('Finite original circuit responses required')
    net_snapshots = np.stack(net_snapshots)
    np.savez_compressed(run/'responses.npz', dnb06=traces, intent=output,
                        dnb06_net_snapshots=net_snapshots)
    reports = []
    for index, condition in enumerate(conditions, 1):
        delta = traces[:, index]-traces[:, 0]
        action = output[:, index]-output[:, 0]
        reports.append({**condition, 'pulse_end_dnb06_delta': delta[74].tolist(),
            'pulse_peak_abs_dnb06': float(np.max(np.abs(delta[50:75]))),
            'post_500ms_dnb06_delta': delta[124].tolist(),
            'pulse_end_output_delta': action[74].tolist(),
            'pulse_peak_abs_steering_readout': float(np.max(np.abs(action[50:75, 0])))})
    if (sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != selected['checkpoint_sha256']
            or sha256(episode) != entry['sha256']):
        raise ValueError('Fixed evidence changed during experiment')
    result = {'status': 'completed_dnb06_synthetic_motion_feature_pulses',
        'conditions': reports, 'responses_sha256': sha256(run/'responses.npz'),
        'net_snapshot_ticks': [50, 75, 125],
        'dnb06_net_min_per_neuron': net_snapshots.min(axis=(0, 1)).tolist(),
        'dnb06_net_max_per_neuron': net_snapshots.max(axis=(0, 1)).tolist(),
        'stimulus_is_not_image_or_measured_fly_response': True,
        'normalization_scale_is_train_only': True, 'direct_ports_fixed': True,
        'closed_loop_or_physiology_verified': False, 'actor_deployed': False,
        'optimizer_updates': 0, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps({'status': result['status'], 'conditions': len(reports)}), flush=True)


if __name__ == '__main__':
    main()
