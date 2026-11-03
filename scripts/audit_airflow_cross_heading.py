"""Compare original unqualified wind candidates across recorded body headings."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.recording) < 2 or len(set(args.recording)) != len(args.recording):
        raise ValueError('At least two distinct original diagnostics required')
    sources, reports = [], []
    for name in args.recording:
        root = Path(name)
        result, meta = read(root/'result.json'), root/'diagnostic-frames.json'
        if result['status'] != 'completed' or sha256(meta) != result['metadata_sha256']:
            raise ValueError('Completed unchanged diagnostic metadata required')
        metadata = read(meta)
        rows = []
        for i, row in enumerate(metadata):
            frame = Path(row['source_file'])
            if row['index'] != i or sha256(frame) != row['sha256']:
                raise ValueError('Original indexed frame/hash differs')
            value = read(frame)
            d = value['airflow_diagnostic']
            if (d != row['diagnostic'] or value['aid_probe_only'] is not True
                    or d['source_verified'] is not False or d['actor_admissible'] is not False):
                raise ValueError('Original diagnostic-only evidence required')
            rows.append(d)
        if len(rows) < 2 or len({r['generation'] for r in rows}) != 1:
            raise ValueError('Same-generation multi-frame evidence required')
        wind = np.asarray([r['wind_velocity_candidate_m_s'] for r in rows])
        velocity = np.asarray([r['vehicle_velocity_candidate_m_s'] for r in rows])
        axes = np.asarray([[r['body_forward_candidate'], r['body_left_candidate'],
                            r['body_up_candidate']] for r in rows])
        weather = np.asarray([r['weather_wind_velocity_candidate_kmh'] for r in rows])
        weather_speed = np.asarray([r['weather_wind_speed_candidate_kmh'] for r in rows])
        if (wind.shape != (len(rows), 3) or velocity.shape != wind.shape
                or axes.shape != (len(rows), 3, 3) or weather.shape != (len(rows), 2)
                or not all(np.isfinite(x).all() for x in (wind, velocity, axes, weather,
                                                          weather_speed))):
            raise ValueError('Finite original candidate vectors required')
        speed = np.linalg.norm(velocity, axis=1)
        moving = speed > 1.
        if not moving.any():
            raise ValueError('Moving body required for each diagnostic')
        forward = axes[:, 0]
        heading = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
        body_wind = np.einsum('nij,nj->ni', axes, wind)
        relative = np.einsum('nij,nj->ni', axes, wind-velocity)
        gram = axes @ np.swapaxes(axes, 1, 2)
        sources.append({'recording': name, 'result_sha256': sha256(root/'result.json'),
                        'metadata_sha256': sha256(meta)})
        reports.append({'frames': len(rows), 'moving_frames': int(moving.sum()),
            'heading_span_degrees': float(np.ptp(heading)*180/np.pi),
            'max_orthogonality_error': float(np.abs(gram-np.eye(3)).max()),
            'determinant_range': [float(np.linalg.det(axes).min()),
                                  float(np.linalg.det(axes).max())],
            'wind_speed_m_s_p0_p50_p100': np.quantile(np.linalg.norm(wind, axis=1),
                                                       [0, .5, 1]).tolist(),
            'weather_speed_kmh_min_max': [float(weather_speed.min()), float(weather_speed.max())],
            'weather_vector_kmh_max_norm': float(np.linalg.norm(weather, axis=1).max()),
            'body_forward_wind_m_s_min_max': [float(body_wind[:, 0].min()),
                                              float(body_wind[:, 0].max())],
            'body_relative_forward_m_s_moving_min_max': [float(relative[moving, 0].min()),
                                                         float(relative[moving, 0].max())],
            'velocity_forward_positive_moving_count': int(
                (np.einsum('ni,ni->n', forward, velocity)[moving] > 0).sum())})
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'sources': sources,
        'scope': 'diagnostic candidate comparison; no wind qualification or actor input'})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    output = {'status': 'completed_cross_heading_candidate_airflow_audit',
              'recordings': reports, 'wind_units_axes_sign_physically_verified': False,
              'wind_independent_stimulus_present': False, 'physics_sync_verified': False,
              'training_admissible': False, 'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
