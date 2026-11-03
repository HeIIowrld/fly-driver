"""Audit candidate body-axis projections and source clocks; never qualify wind."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result_path, metadata_path = root / 'result.json', root / 'diagnostic-frames.json'
    result = json.loads(result_path.read_text(encoding='utf8'))
    if result['status'] != 'completed' or sha256(metadata_path) != result['metadata_sha256']:
        raise ValueError('Completed recording with unchanged metadata required')
    rows = json.loads(metadata_path.read_text(encoding='utf8'))
    original_hashes = {str(p): sha256(p) for p in (result_path, metadata_path)}
    data = []
    for row in rows:
        source = Path(row['source_file'])
        if sha256(source) != row['sha256']:
            raise ValueError('Recorded source frame changed')
        frame = json.loads(source.read_text(encoding='utf8'))
        d = frame['airflow_diagnostic']
        if (d != row['diagnostic'] or frame['aid_probe_only'] is not True
                or d['actor_admissible'] is not False or d['source_verified'] is not False):
            raise ValueError('Only original unqualified diagnostics are accepted')
        data.append(d)
    if not data or len({d['generation'] for d in data}) != 1:
        raise ValueError('One nonempty original generation required')
    velocity = np.asarray([d['vehicle_velocity_candidate_m_s'] for d in data])
    wind = np.asarray([d['wind_velocity_candidate_m_s'] for d in data])
    axes = np.asarray([[d['body_forward_candidate'], d['body_left_candidate'],
                        d['body_up_candidate']] for d in data])
    starts = np.asarray([d['started_source_clock'] for d in data])
    ends = np.asarray([d['finished_source_clock'] for d in data])
    physics = np.asarray([d['car_physics_timestamp_ms'] for d in data])
    if (velocity.shape != (len(data), 3) or wind.shape != velocity.shape
            or axes.shape != (len(data), 3, 3)
            or not all(np.isfinite(x).all() for x in (velocity, wind, axes, starts, ends, physics))
            or np.any(ends < starts) or np.any(np.diff(starts) <= 0)):
        raise ValueError('Finite, ordered candidate observations required')
    speed = np.linalg.norm(velocity, axis=1)
    wind_speed = np.linalg.norm(wind, axis=1)
    weather_speed = np.asarray([d['weather_wind_speed_candidate_kmh'] for d in data])
    if weather_speed.shape != (len(data),) or not np.isfinite(weather_speed).all():
        raise ValueError('Finite candidate weather wind speed required')
    moving = speed > 1.0
    if not np.any(moving):
        raise ValueError('Moving samples required')
    body_velocity = np.einsum('nij,nj->ni', axes, velocity)
    candidate_relative = np.einsum('nij,nj->ni', axes, wind - velocity)
    if any(sha256(path) != digest for path, digest in original_hashes.items()):
        raise ValueError('Original evidence changed during audit')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
                                  'source_hashes': original_hashes,
                                  'motion_threshold_m_s': 1.0, 'actor_input': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    out = {
        'status': 'completed_candidate_airflow_axis_audit',
        'frames': len(data), 'moving_frames': int(moving.sum()),
        'moving_forward_velocity_m_s_range': [float(body_velocity[moving, 0].min()),
                                               float(body_velocity[moving, 0].max())],
        'moving_forward_nonpositive_count': int((body_velocity[moving, 0] <= 0).sum()),
        'wind_speed_m_s_quantiles_min_median_p95_max': np.quantile(
            wind_speed, [0, .5, .95, 1]).tolist(),
        'wind_to_vehicle_speed_ratio_moving_median_p95_max': np.quantile(
            wind_speed[moving]/speed[moving], [.5, .95, 1]).tolist(),
        'weather_wind_speed_kmh_range': [float(weather_speed.min()), float(weather_speed.max())],
        'wind_as_independent_training_signal_qualified': False,
        'moving_abs_lateral_to_speed_median': float(np.median(
            np.abs(body_velocity[moving, 1]) / speed[moving])),
        'candidate_relative_front_m_s_range': [float(candidate_relative[moving, 0].min()),
                                                float(candidate_relative[moving, 0].max())],
        'source_read_span_ms_max': float(np.max(ends - starts) * 1000),
        'source_clock_step_ms_min': float(np.min(np.diff(starts)) * 1000),
        'car_timestamp_candidate_step_ms_min': float(np.min(np.diff(physics))),
        'nonincreasing_car_timestamp_steps': int((np.diff(physics) <= 0).sum()),
        'car_timestamp_is_physics_sample_time_verified': False,
        'first_last_wind_vector_difference_m_s': float(np.linalg.norm(wind[-1] - wind[0])),
        'wind_source_or_physics_synchronization_verified': False,
        'relative_wind_is_diagnostic_only': True, 'actor_deployed': False,
        'goal_achieved': False,
    }
    write_json(run / 'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
