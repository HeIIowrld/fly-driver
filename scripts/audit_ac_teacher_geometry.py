"""Recompute privileged geometry from public poses; no game or fly actor access."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_sac_teacher import (
    PrivilegedSACTeacher,
    TeacherReferenceCurve,
    encode_teacher_observation,
)
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.numeric_pickle import load_numeric_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    record = Path('assets/external/acgym-lap-sample-20261027/raw_data.pkl')
    config = json.loads(Path('configs/ac-sac-replay.json').read_text())
    if sha256(record) != config['record_sha256']:
        raise ValueError('Published record hash differs')
    track = Path('assets/external/ac-teacher-geometry-20261027/monza.csv')
    reference = Path('assets/external/acgym-teacher-recipe-20261027/legacy'
                     '/assetto_corsa_gym/AssettoCorsaConfigs/tracks/monza-racing_line.csv')
    run = begin_run(args.output, dict(scope=__doc__, record_sha256=sha256(record),
        track_sha256=sha256(track), reference_sha256=sha256(reference)))
    geometry = PrivilegedTrackGeometry(track, reference, track_sha256=sha256(track),
                                       reference_sha256=sha256(reference))
    states = load_numeric_pickle(record.read_bytes())['states']
    gaps, rays, rebuilt = [], [], []
    for row in states:
        values = geometry.features(row['world_position_x'], row['world_position_y'], row['yaw'])
        gaps.append(abs(values['gap'] - row['gap']))
        rays.append(np.max(np.abs(values['sensors'] - row['sensors'])))
        rebuilt.append(dict(row, **values))
    curve = TeacherReferenceCurve(reference)
    observations = np.array([encode_teacher_observation(
        row, rebuilt[:index], curve.segment(row['LapDist']), profile=config['profile'])
        for index, row in enumerate(rebuilt) if index >= 3])
    cached_root = Path('runs/ac-teacher-replay-20261027')
    relative = 't21b/runs/teacher-replay/privileged-teacher-replay.npz'
    cached = cached_root / relative
    receipt = json.loads((cached_root / 'collection-receipt.json').read_text())
    if sha256(cached) != receipt['files_sha256'][relative]:
        raise ValueError('Previously verified teacher replay differs')
    teacher = PrivilegedSACTeacher(config['checkpoint'], config['checkpoint_sha256'])
    actions = teacher.predict(observations)['deterministic_policy_action']
    with load_array(cached, allow_pickle=False) as data:
        observation_error = float(np.abs(observations - data['teacher_observations']).max())
        action_error = float(np.abs(actions - data['teacher_policy_delta_action']).max())
    latency, sequential_actions = [], []
    for index, row in enumerate(states):
        if index < 3:
            continue
        started = time.perf_counter()
        values = geometry.features(row['world_position_x'], row['world_position_y'], row['yaw'])
        observation = encode_teacher_observation(dict(row, **values), rebuilt[:index],
            curve.segment(row['LapDist']), profile=config['profile'])
        sequential_actions.append(teacher.predict(observation[None])['deterministic_policy_action'][0])
        latency.append(time.perf_counter() - started)
    np.savez_compressed(run / 'privileged-recomputed-teacher.npz', observations=observations,
                        teacher_delta_actions=actions)
    result = dict(rows=len(states), max_gap_absolute_error=float(max(gaps)),
        max_ray_absolute_error=float(max(rays)),
        all_recorded_features_matched=bool(max(gaps) < 1e-8 and max(rays) < 1e-8),
        teacher_observation_max_absolute_error=observation_error,
        teacher_action_max_absolute_error=action_error,
        teacher_recomputed_rows=len(actions),
        single_observation_compute_seconds=dict(
            p50=float(np.percentile(latency, 50)), p95=float(np.percentile(latency, 95)),
            p99=float(np.percentile(latency, 99)), max=float(max(latency))),
        sequential_batch_action_max_error=float(np.abs(sequential_actions - actions).max()),
        timing_scope='offline pose -> geometry ->125obs ->SAC; excludes game/capture/transport; '
                     'one CPU thread, not hard real-time',
        reconstruction_checks_passed=bool(max(gaps) < 1e-8 and max(rays) < 1e-8
            and observation_error < 1e-12 and action_error < 1e-5),
        actor_truth_inputs_added=False, teacher_game_driving_verified=False, goal_achieved=False)
    write_json(run / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
