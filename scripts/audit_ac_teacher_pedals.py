"""Compare recorded offline teacher requests with fly output codec limits; no game input."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_sac_teacher import legacy_request_to_normalized_controls
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collected', default='runs/ac-teacher-replay-20261027')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.collected)
    relative = 't21b/runs/teacher-replay/privileged-teacher-replay.npz'
    replay = root / relative
    receipt = json.loads((root / 'collection-receipt.json').read_text())
    if sha256(replay) != receipt['files_sha256'][relative]:
        raise ValueError('Teacher replay differs from verified collection')
    run = begin_run(args.output, dict(scope=__doc__, replay_sha256=sha256(replay)))
    with load_array(replay, allow_pickle=False) as data:
        controls = np.array([legacy_request_to_normalized_controls(row)
                             for row in data['suggested_absolute_request']])
    if not len(controls):
        raise ValueError('No teacher requests')
    pedals = controls[:, 1:]
    excess = np.maximum(pedals.sum(1) - 1, 0)
    np.savez_compressed(run / 'normalized-teacher-requests.npz', requests=controls)
    write_json(run / 'result.json', dict(
        status='completed', requests=len(controls),
        normalized_min=controls.min(0).tolist(), normalized_max=controls.max(0).tolist(),
        throttle_brake_sum_max=float(pedals.sum(1).max()),
        requests_outside_legacy_attenuated_domain=int((excess > 1e-6).sum()),
        legacy_coupling_best_possible_per_pedal_rmse_lower_bound=float(
            np.sqrt(np.mean(excess ** 2) / 4)),
        requests_above_gain_one_independent_caps=(pedals > -np.expm1(-1)).sum(0).tolist(),
        normalized_requests_sha256=sha256(run / 'normalized-teacher-requests.npz'),
        source='pinned car_control.py maps each pedal with (request+1)/2 independently',
        source_registry='assets/external/ac-control-codec-20261027/source.json',
        scope='offline teacher suggested requests, not actual game-applied values',
        vision_paired=False, model_training_updates=0, actuator_messages_sent=0,
        goal_achieved=False))


if __name__ == '__main__':
    main()
