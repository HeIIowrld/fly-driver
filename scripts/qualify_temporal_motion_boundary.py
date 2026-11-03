"""Qualify a neutral temporal motion boundary on one saved TRAIN prefix, without fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    training = Path(a.training)
    tr, tm = read(training/'result.json'), read(training/'manifest.json')
    plan = tm['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'train']
    if (tr['status'] != 'completed_fixed_joint_command_training'
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or sha256(source['checkpoint']) != source['checkpoint_sha256']
            or len(entries) != 6 or not torch.cuda.is_available()):
        raise ValueError('Exact fixed graph, original TRAIN roles and GPU required')
    entry = entries[0]
    path = root/entry['file']
    if sha256(path) != entry['sha256']:
        raise ValueError('Original TRAIN history changed')
    names = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')
    with load_array(path, allow_pickle=False) as x:
        data = {key: x[key][:64].copy() for key in names}
        times = x['observation_time'][:64].copy()
    complete_history(times, times)
    if len(times) != 64 or data['visual_valid'].any():
        raise ValueError('64 original TRAIN observations and original mask required')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
        source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    cp = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)
    if cp['schema'] != model.schema or cp['spec'] != model.spec:
        raise ValueError('Original model specification required')
    model.load_state_dict(cp['model'], strict=True)
    model.eval().requires_grad_(False)
    boundary = TemporalMotionBoundary(model.backbone.scene_command.mean,
        model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to('cuda')
    t = {key: torch.tensor(data[key], device='cuda') for key in names}
    with torch.no_grad():
        original = model._drive_tensor(*(t[key] for key in names))
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'training_checkpoint_sha256': tr['checkpoint_sha256'],
        'episode_sha256': entry['sha256'], 'dataset_sha256': sha256(root/'dataset.json'),
        'boundary_source_sha256': sha256('src/flybrain_drive/flight_temporal_motion_boundary.py'),
        'observations': 64, 'role': 'original_train_prefix',
        'time_constants_seconds': [.05, .2], 'units': 'engineering hypothesis',
        'target_used': False, 'optimizer_updates': 0, 'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    state, reference_state, traces, held, reference_held = (
        model.initial_state(1), model.initial_state(1), None, None, None)
    outputs, reference_outputs, norms = [], [], []
    for index in range(len(times)):
        elapsed = None if index == 0 else float(times[index]-times[index-1])
        if index:
            for dt in substeps(elapsed):
                state = model._advance_elapsed_tensor(held, state, dt)
                with torch.no_grad():
                    reference_state = model._advance_elapsed_tensor(reference_held,
                                                                    reference_state, dt)
        outputs.append(model._read_state_tensor(state)[0])
        with torch.no_grad():
            reference_outputs.append(model._read_state_tensor(reference_state)[0])
        delta, traces = boundary.step(t['scene_features'][index:index+1],
            torch.ones((1, 4), dtype=torch.bool, device='cuda'), traces, elapsed)
        norms.append(float((traces[0]-traces[1]).norm().detach()))
        held = boundary.modulate(original[index:index+1], model.backbone.vision_ports, delta)
        reference_held = original[index:index+1]
    output = torch.cat(outputs)
    reference = torch.cat(reference_outputs)
    error = float((output-reference).abs().max().detach())
    if error > 1e-6 or max(norms) <= 0:
        raise ValueError('Neutral model parity and nonconstant original traces required')
    loss = output[32:].square().mean()
    loss.backward()
    grad = boundary.projection.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
        raise ValueError('Finite nonzero temporal-boundary gradient through actual graph required')
    np.savez_compressed(run/'qualification.npz', time=times,
        neutral_output=output.detach().cpu().numpy(), reference_output=reference.cpu().numpy(),
        fast_slow_norm=np.asarray(norms), gradient=grad.detach().cpu().numpy())
    result = {'status': 'qualified_neutral_temporal_motion_graph', 'observations': len(times),
        'intervals': len(times)-1, 'baseline_max_abs_error': error,
        'fast_slow_norm_max': max(norms), 'gradient_norm': float(grad.norm()),
        'qualification_sha256': sha256(run/'qualification.npz'),
        'causal_prior_held_input': True, 'target_used': False, 'optimizer_updates': 0,
        'physiology_verified': False, 'physical_time_sync_verified': False,
        'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
