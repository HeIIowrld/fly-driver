"""Compare original diagnostic airflow frames with same-generation AI telemetry."""

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
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result = read(root/'result.json')
    if (result['status'] != 'completed' or sha256(root/'diagnostic-frames.json')
            != result['metadata_sha256']):
        raise ValueError('Completed diagnostic with exact metadata required')
    rows = read(root/'diagnostic-frames.json')
    receipt = read(root/'control/api-receipt.json')
    if receipt['status'] != 'api_returned_true' or not receipt['motion_probe_passed']:
        raise ValueError('Owned motion receipt required')
    telemetry = [row for row in receipt['telemetry'] if row['phase'] == 'driving'
                 and row['ai_controlled'] and not row['paused']]
    clocks = np.array([x['script_clock'] for x in telemetry])
    if len(clocks) < 2 or np.any(np.diff(clocks) <= 0):
        raise ValueError('Original increasing AI source clocks required')
    hashes = {str(p): sha256(p) for p in (root/'result.json', root/'diagnostic-frames.json',
              root/'control/api-receipt.json', root/'control/result.json')}
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': hashes,
        'maximum_source_clock_pair_gap_seconds': .03, 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    all_wind, weather, paired, gaps = [], [], [], []
    speed_errors, determinants, orthogonal = [], [], []
    for row in rows:
        path = Path(row['source_file'])
        if sha256(path) != row['sha256']:
            raise ValueError('Original frame changed')
        frame = read(path)
        diag = frame['airflow_diagnostic']
        if (diag != row['diagnostic'] or not frame['aid_probe_only']
                or diag['source_verified'] is not False or diag['actor_admissible'] is not False):
            raise ValueError('Original unqualified diagnostic changed')
        wind = np.asarray(diag['wind_velocity_candidate_m_s'])
        velocity = np.asarray(diag['vehicle_velocity_candidate_m_s'])
        axes = np.asarray([diag['body_forward_candidate'], diag['body_left_candidate'],
                           diag['body_up_candidate']])
        if not all(np.isfinite(x).all() for x in (wind, velocity, axes)):
            raise ValueError('Finite original candidate values required')
        all_wind.append(float(np.linalg.norm(wind)))
        weather.append(diag['weather_wind_speed_candidate_kmh'])
        determinants.append(float(np.linalg.det(axes)))
        orthogonal.append(float(np.max(np.abs(axes@axes.T-np.eye(3)))))
        middle = (diag['started_source_clock']+diag['finished_source_clock'])/2
        nearest = int(np.argmin(np.abs(clocks-middle)))
        gap = abs(float(clocks[nearest]-middle))
        if gap <= .03:
            gap_error = abs(float(np.linalg.norm(velocity)*3.6
                                  - telemetry[nearest]['speed_kmh']))
            paired.append(row['index'])
            gaps.append(gap)
            speed_errors.append(gap_error)
    if not paired:
        raise ValueError('No time-near motion samples; no interpolation or invented pairing')
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Original evidence changed')
    out = {'status': 'completed_airflow_motion_source_audit', 'frames': len(rows),
        'paired_driving_frames': len(paired), 'maximum_pair_gap_seconds': max(gaps),
        'candidate_vehicle_speed_kmh_median_abs_difference': float(np.median(speed_errors)),
        'candidate_vehicle_speed_kmh_max_abs_difference': max(speed_errors),
        'wind_candidate_m_s_range': [min(all_wind), max(all_wind)],
        'weather_wind_speed_kmh_range': [min(weather), max(weather)],
        'basis_determinant_range': [min(determinants), max(determinants)],
        'maximum_basis_orthogonality_error': max(orthogonal),
        'matching_is_nearest_source_clock_not_same_physics_sample': True,
        'wind_semantics_or_exposure_synchronization_verified': False,
        'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
