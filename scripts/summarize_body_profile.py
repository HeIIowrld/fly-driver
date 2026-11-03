"""Summarize an owned cProfile replay without interpreting it as GPU kernel time."""

import argparse
import pstats
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    stats = pstats.Stats(args.profile)
    targets = {
        ('flight_visual_body_temporal_context.py', 'decision'),
        ('flight_visual_body_airflow.py', 'drive_observed'),
        ('flight_visual_body_airflow.py', 'advance'),
        ('flight_opponent_temporal_boundary.py', 'step'),
        ('flight_temporal_motion_boundary.py', 'step'),
        ('vehicle_circuit.py', '_advance_tensor'),
        ('flight_commands.py', 'assert_body_boundary_protected'),
    }
    rows = []
    for (file, line, name), (primitive, calls, self_s, cumulative_s, _) in stats.stats.items():
        if (Path(file).name, name) in targets:
            rows.append({'file': Path(file).name, 'line': line, 'name': name,
                         'primitive_calls': primitive, 'calls': calls, 'self_seconds': self_s,
                         'inclusive_seconds': cumulative_s})
    if len(rows) != len(targets):
        raise ValueError('Expected original profile call sites not all observed')
    rows.sort(key=lambda row: row['inclusive_seconds'], reverse=True)
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'profile_sha256': sha256(args.profile),
        'recording_result_sha256': sha256(root/'result.json'),
        'recording_program_sha256': sha256(root/'executed-program.py'),
        'seed': None, 'scope': 'cProfile host inclusive times over three replay streams; nested'})
    write_json(run/'result.json', {'status': 'completed_body_host_profile_summary', 'sites': rows,
        'full_profile_seconds': stats.total_tt, 'live_contention_tested': False,
        'gpu_kernel_occupancy_measured': False, 'profiling_overhead_unmeasured': True,
        'training_admissible': False, 'vehicle_control_sent': False})
    for row in rows:
        print(row['file'], row['name'], row['calls'], round(row['inclusive_seconds'], 4))


if __name__ == '__main__':
    main()
