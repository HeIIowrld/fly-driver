"""Offline observed-history value interventions at the physical motor input boundary."""

import argparse
import json

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--body', required=True)
    parser.add_argument('--body-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.body) != args.body_sha256:
        raise ValueError('Original replay body hash mismatch')
    hashes = {p: sha256(p) for p in (args.spec, args.checkpoint, args.body)}
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    motor = PhysicalManeuverCircuit(args.spec, payload, sparse_backend='cached_csr_cpu').eval()
    with np.load(args.body, allow_pickle=False) as data:
        values, valid = data['values'].copy(), data['valid'].copy()
        names = data['feature_names'].tolist()
    if values.shape != (986, 21) or valid.shape != values.shape or not valid[:, [4, 16]].all():
        raise ValueError('Expected original complete 986-state body21')
    run = begin_run(args.output, {'source_hashes': hashes, 'program_sha256': sha256(__file__),
        'role': 'offline_fixed_command_input_boundary_only', 'actor_deployed': False})
    body = torch.tensor(values, dtype=torch.float32)
    mask = torch.tensor(valid, dtype=torch.bool)
    command = torch.tensor([[0., 30/3.6]], dtype=torch.float32).repeat(len(body), 1)
    goals = torch.tensor([[0., 1.]], dtype=torch.float32).repeat(len(body), 1)
    kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    reports = []
    with torch.no_grad():
        original = motor.drive(command, body, mask, goals, **kwargs)
        for index, name in enumerate(names):
            changed = body.clone()
            # Perturb available rows using the same observed channel distribution.
            indices = torch.where(mask[:, index])[0]
            if len(indices) > 1:
                changed[indices, index] = body[indices.roll(1), index]
            modified = motor.drive(command, changed, mask, goals, **kwargs)
            delta = (modified-original).abs()
            reports.append({'index': index, 'name': name, 'valid_rows': int(len(indices)),
                'changed_rows': int((changed[:, index] != body[:, index]).sum()),
                'max_abs_drive_difference': float(delta.max()),
                'mean_abs_drive_difference': float(delta.mean())})
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise ValueError('Input changed during probe')
    result = {'status': 'completed', 'reports': reports, 'rows': len(body),
        'fixed_desired_yaw_rad_s': 0, 'fixed_desired_speed_m_s': 30/3.6,
        'intervention': 'cyclic permutation within available rows of one observed channel',
        'source_hashes': hashes, 'visual_and_recurrent_state_not_evaluated': True,
        'actor_deployed': False, 'optimizer_updates': 0, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
