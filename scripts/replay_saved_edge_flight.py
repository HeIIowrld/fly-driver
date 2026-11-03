"""Replay saved author FlyVis central-cell edge responses through the flight graph.

A central response is replicated across spatial bins: this is a synthetic spatial
intervention, not a reconstruction of a real camera or a full retinal response.
"""
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
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'candidate', 'source', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    training, candidate, source = map(Path, (a.training, a.candidate, a.source))
    receipt, manifest, saved = (read(training/'result.json'), read(training/'manifest.json'),
                                read(source/'result.json'))
    plan = manifest['config']['plan']
    original = read(Path(plan['source'])/'result.json')
    trained = read(candidate/'result.json')
    if (receipt['status'] != 'completed_fixed_joint_command_training'
            or sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']
            or sha256(original['checkpoint']) != original['checkpoint_sha256']
            or trained['status'] != 'completed_one_pass_hsvs_boundary_train'
            or trained['source_checkpoint_sha256'] != receipt['checkpoint_sha256']
            or sha256(candidate/'visual-boundary.pt') != trained['checkpoint_sha256']
            or saved['status'] != 'completed'
            or sha256(source/'responses.npz') != saved['responses_sha256']):
        raise ValueError('Exact completed source and checkpoints required')
    stimulus = read(source/'stimulus.json')
    if stimulus['dt'] != .005 or stimulus['mode'] != 'edge':
        raise ValueError('Original edge time base required')
    parameters = stimulus['parameters']
    selected = [next(i for i, v in enumerate(parameters) if v['angle'] == angle
                     and v['intensity'] == 1 and v['speed'] == 19.0)
                for angle in (0, 180)]
    with load_array(source/'responses.npz', allow_pickle=False) as x:
        names = x['central_cell_types'].tolist()
        indices = [names.index(f'{t}{d}') for t in ('T4', 'T5') for d in 'abcd']
        length = min(int(x['valid_frames'][i]) for i in selected)
        activity = x['activity'][selected, :length, :][:, :, indices].copy()
    if not np.isfinite(activity).all() or length < 2:
        raise ValueError('Finite original response histories required')
    root = Path(plan['data'])
    entries = [e for e in read(root/'dataset.json')['episodes'] if e['role'] == 'train']
    entry = entries[0]
    if sha256(root/entry['file']) != entry['sha256']:
        raise ValueError('Body history hash mismatch')
    with load_array(root/entry['file'], allow_pickle=False) as x:
        data = {k: x[k][:1].copy() for k in
                ('visual', 'visual_valid', 'body', 'body_valid', 'goals')}
    if data['visual_valid'].any() or not torch.cuda.is_available():
        raise ValueError('Expected original visual mask and CUDA required')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'],
        torch.load(original['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    checkpoint = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['spec'] != model.spec or checkpoint['schema'] != model.schema:
        raise ValueError('Graph mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    boundary = torch.load(candidate/'visual-boundary.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary['weight'].to('cuda'))
        model.backbone.vision.bias.copy_(boundary['bias'].to('cuda'))
    spec = read(plan['source_spec'])
    with load_array(spec['graph_file'], allow_pickle=False) as graph:
        ids = np.flatnonzero(graph['types'] == 'DNb06')
    if len(ids) != 2:
        raise ValueError('Expected two DNb06 neurons')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'source_result_sha256': sha256(source/'result.json'),
        'source_stimulus_sha256': sha256(source/'stimulus.json'),
        'source_responses_sha256': saved['responses_sha256'],
        'candidate_checkpoint_sha256': trained['checkpoint_sha256'],
        'body_episode_sha256': entry['sha256'], 'body_index': 0,
        'stimulus_indices': selected, 'angles': [0, 180], 'dt': .005,
        'spatial_assumption': 'one central neuron replicated in all 16 bins per type',
        'direct_motion_ports': 'fixed at train mean', 'actor_deployed': False,
        'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    t = {k: torch.tensor(v, device='cuda').expand(9, *v.shape[1:]) for k, v in data.items()}
    mean = model.backbone.scene_command.mean
    direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    state = model.initial_state(9)
    outputs, neural = [], []
    with torch.no_grad():
        for frame in range(length):
            scene = mean.expand(9, -1).clone().reshape(9, 4, 9, 16)
            # Rows 0..3 and 4..7 are angles 0 and 180, row 8 is train mean.
            for k in range(2):
                for view in range(4):
                    scene[k*4+view, view, 1:, :] = torch.tensor(
                        np.tanh(activity[k, frame]), device='cuda')[:, None]
            scene = scene.flatten(1)
            drive = model._drive_tensor(*(t[k] for k in
                ('visual', 'visual_valid', 'body', 'body_valid', 'goals')), scene)
            reference = model._drive_tensor(*(t[k] for k in
                ('visual', 'visual_valid', 'body', 'body_valid', 'goals')),
                mean.expand(9, -1))
            drive[:, direct] = reference[:, direct]
            if not torch.equal(drive[:, direct], reference[:, direct]):
                raise ValueError('Direct ports changed')
            for dt in substeps(.005):
                state = model._advance_elapsed_tensor(drive, state, dt)
            outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
            neural.append(state[:, ids].cpu().numpy())
    outputs, neural = np.stack(outputs), np.stack(neural)
    np.savez_compressed(run/'replay.npz', outputs=outputs, dnb06=neural,
                        central_activity=activity)
    summary = {}
    for j, angle in enumerate((0, 180)):
        for v, view in enumerate(('front', 'left', 'right', 'rear')):
            row = j*4+v
            delta = outputs[:, row]-outputs[:, 8]
            dn = neural[:, row]-neural[:, 8]
            summary[f'{angle}_{view}'] = {'output_max_abs': np.max(np.abs(delta), axis=0).tolist(),
                'dnb06_max_abs': np.max(np.abs(dn), axis=0).tolist()}
    write_json(run/'result.json', {'status': 'completed_saved_edge_flight_replay',
        'frames': length, 'conditions': summary, 'replay_sha256': sha256(run/'replay.npz'),
        'synthetic_spatial_replication': True, 'real_driving_validated': False,
        'physiology_validated': False, 'actor_deployed': False})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
