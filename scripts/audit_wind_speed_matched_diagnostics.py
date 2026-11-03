"""Compare unqualified airflow candidates in two independent diagnostic recordings."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def load_recording(root):
    root = Path(root)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    metadata = root/'diagnostic-frames.json'
    if result.get('status') != 'completed' or sha256(metadata) != result.get('metadata_sha256'):
        raise ValueError('Completed original metadata required')
    rows = json.loads(metadata.read_text(encoding='utf8'))
    values = []
    for index, row in enumerate(rows):
        source = Path(row['source_file'])
        if row['index'] != index or sha256(source) != row['sha256']:
            raise ValueError('Original frame identity differs')
        frame = json.loads(source.read_text(encoding='utf8'))
        d = frame['airflow_diagnostic']
        if (d != row['diagnostic'] or frame.get('aid_probe_only') is not True
                or d.get('source_verified') is not False or d.get('actor_admissible') is not False):
            raise ValueError('Only original unqualified diagnostic frames allowed')
        axes = np.array([d['body_forward_candidate'], d['body_left_candidate'],
                         d['body_up_candidate']], dtype=float)
        wind = np.array(d['wind_velocity_candidate_m_s'], dtype=float)
        velocity = np.array(d['vehicle_velocity_candidate_m_s'], dtype=float)
        if (axes.shape != (3, 3) or wind.shape != (3,) or velocity.shape != (3,)
                or not all(np.isfinite(x).all() for x in (axes, wind, velocity))):
            raise ValueError('Finite body axes and candidate vectors required')
        values.append((float(np.linalg.norm(velocity)), axes @ (wind-velocity),
                       d['weather_wind_speed_candidate_kmh']))
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wind', required=True)
    p.add_argument('--zero', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    wind, zero = load_recording(args.wind), load_recording(args.zero)
    if any(v[2] != 35 for v in wind) or any(v[2] != 0 for v in zero):
        raise ValueError('Original weather conditions differ from declared comparison')
    zero_moving = [v for v in zero if v[0] > 1]
    pairs = []
    for v in wind:
        if v[0] <= 1 or not zero_moving:
            continue
        match = min(zero_moving, key=lambda z: abs(v[0]-z[0]))
        if abs(v[0]-match[0]) <= .2:
            pairs.append((v, match))
    if not pairs:
        raise ValueError('No moving candidates matched within 0.2 m/s')
    differences = np.stack([a[1]-b[1] for a, b in pairs])
    run = begin_run(args.output, {'arguments': vars(args), 'program_sha256': sha256(__file__),
        'wind_result_sha256': sha256(Path(args.wind)/'result.json'),
        'zero_result_sha256': sha256(Path(args.zero)/'result.json'), 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    output = {'status': 'completed_diagnostic_speed_matching', 'matched_frames': len(pairs),
        'unique_zero_samples_used': len({id(b) for _, b in pairs}),
        'max_speed_gap_m_s': max(abs(a[0]-b[0]) for a, b in pairs),
        'body_relative_difference_m_s_median': np.median(differences, axis=0).tolist(),
        'body_relative_difference_m_s_min': differences.min(axis=0).tolist(),
        'body_relative_difference_m_s_max': differences.max(axis=0).tolist(),
        'matched_samples_independent': False, 'same_physics_time_verified': False,
        'wind_source_verified': False, 'training_admissible': False}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
