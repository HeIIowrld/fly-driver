"""Qualify synthetic airflow through the real fixed flight graph with recorded body."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_airflow_feedback import ExperimentalAirflowBodyCircuit
from flybrain_drive.graph import sha256
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-plan', required=True)
    parser.add_argument('--fit-result', required=True)
    parser.add_argument('--pedal-checkpoint', required=True)
    parser.add_argument('--body', required=True)
    parser.add_argument('--body-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.body) != args.body_sha256:
        raise ValueError('Original derived body21 changed')
    plan, fit_result = read(args.fit_plan), read(args.fit_result)
    pedal = read(args.pedal_checkpoint)
    if (fit_result['status'] != 'completed_one_pass_pedal_component_fit'
            or fit_result['optimizer_updates'] != 123
            or pedal['schema'] != 'flybrain.two_DNg_final_candidate.v1'
            or pedal['bindings'] != fit_result['bindings']):
        raise ValueError('Original completed two-scalar fit required')
    pins = {p: sha256(p) for p in
            (args.fit_plan, args.fit_result, args.pedal_checkpoint, args.body)}
    with np.load(args.body, allow_pickle=False) as a:
        values, valid = a['values'].copy(), a['valid'].copy()
        clocks = a['decision_monotonic_seconds'].copy()
    if (values.shape != (986, 21) or valid.shape != values.shape or valid.dtype != bool
            or not np.isfinite(values).all() or not valid[:, [4, 16]].all()
            or np.any((np.diff(clocks) <= 0) | (np.diff(clocks) > .08))):
        raise ValueError('Complete original body values, masks and clocks required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': pins,
        'role': 'synthetic_airflow_through_actual_graph_qualification', 'optimizer_updates': 0,
        'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    original = OriginalGraphAdapter(plan, lambda: None)
    with torch.no_grad():
        for name, parameter in original.parameters().items():
            parameter.copy_(torch.tensor(pedal['values'][name], device=parameter.device))
        quiet = original.initialize()
    expected_fingerprint = fit_result['fit']['final_fixed_rollout']['fixed_fingerprint']
    if original.fixed_fingerprint() != expected_fingerprint:
        raise ValueError('Original fixed graph/readout fingerprint mismatch')
    original.model.requires_grad_(False)
    wrapped = ExperimentalAirflowBodyCircuit(original.model).to('cuda').eval()
    body = torch.tensor(values, dtype=torch.float32, device='cuda')
    mask = torch.tensor(valid, dtype=torch.bool, device='cuda')
    command = torch.tensor([[0., 30/3.6]], device='cuda').repeat(len(body), 1)
    goals = torch.tensor([[0., 1.]], device='cuda').repeat(len(body), 1)
    airflow = torch.tensor([[3., 5., -1.]], device='cuda').repeat(len(body), 1)
    airflow_valid = torch.ones_like(airflow, dtype=torch.bool)
    with torch.no_grad():
        original_drive = original.model.drive(command, body, mask, goals, **original.kwargs)
        neutral_drive = wrapped.drive(command, body, mask, goals, airflow, airflow_valid,
                                      **original.kwargs)
    if not torch.equal(original_drive, neutral_drive):
        raise ValueError('Zero adapter must retain exact original drive for every recorded state')
    state = quiet.detach()
    for i in range(8):
        elapsed = float(clocks[i+1]-clocks[i])
        _, _, state = wrapped.advance(command[i:i+1], body[i:i+1], mask[i:i+1],
            goals[i:i+1], airflow[i:i+1], airflow_valid[i:i+1], state, elapsed,
            **original.kwargs)
    prediction = original.predict(state)['original_intent']
    if not torch.isfinite(prediction).all():
        raise ValueError('Finite actual-graph readout required')
    prediction.square().sum().backward()
    grad = wrapped.airflow_adapter.projection.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
        raise ValueError('Finite nonzero gradient through graph to sensory boundary required')
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('Original evidence changed during qualification')
    result = {'status': 'qualified_synthetic_airflow_actual_graph',
        'synthetic_airflow_body_frame_m_s': [3., 5., -1.],
        'actual_airflow_recorded': False, 'original_body_observations': len(body),
        'neutral_drive_max_difference': 0,
        'actual_intervals_replayed_for_gradient': 8,
        'gradient_norm': float(grad.norm()), 'valid_counts': mask.sum(0).cpu().tolist(),
        'graph_and_original_parameters_frozen': True, 'optimizer_updates': 0,
        'sensor_mapping_is_engineering_not_measured_fly_physiology': True,
        'braking_or_closed_loop_verified': False, 'actor_deployed': False,
        'wall_seconds': time.perf_counter()-started, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
