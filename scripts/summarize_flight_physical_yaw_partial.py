"""Report the actual first physical pair without qualifying an incomplete or unmatched study."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-audit', required=True)
    p.add_argument('--zero-audit', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    full_path, zero_path = Path(a.full_audit), Path(a.zero_audit)
    full, zero = read(full_path / 'result.json'), read(zero_path / 'result.json')
    plan = read(a.plan)
    for value in (full, zero):
        assert value['status'] == 'completed_physical_yaw_trial_audit'
        assert value['observations'] == 257 and value['trial']['sign'] == 1
        assert value['original_full_history_replayed'] and value['intervals_above_80ms'] == 0
    assert full['trial']['mode'] == 'full' and zero['trial']['mode'] == 'zero_yaw'
    failures = [
        'runs/flight-physical-yaw-block0-20261029-t1161/trial-00',
        'runs/flight-physical-yaw-block0-20261029-t1162/trial-01',
        'runs/flight-physical-yaw-block0-continuation-20261029-t1164/trial-01',
        'runs/flight-physical-yaw-block0-resume-20261029-t1166/trial-01',
        'runs/flight-physical-yaw-block0-timing-20261029-t1168/trial-02']
    run = begin_run(a.output, dict(arguments=vars(a), plan_sha256=sha256(a.plan),
        full_audit_sha256=sha256(full_path / 'result.json'),
        zero_audit_sha256=sha256(zero_path / 'result.json'),
        timing_amendment='docs/flight-physical-yaw-timing-amendment-20261029.md',
        timing_amendment_sha256=sha256('docs/flight-physical-yaw-timing-amendment-20261029.md'),
        actuator_messages_sent=0))
    width_full = full['pulse_off_host_clock']-full['pulse_on_host_clock']
    width_zero = zero['pulse_off_host_clock']-zero['pulse_on_host_clock']
    spec = plan['evaluation']
    differences = dict(start_position_m=float(np.linalg.norm(np.array(full['initial_position'])
                      - zero['initial_position'])),
        baseline_speed_kmh=abs(full['baseline_speed_kmh']-zero['baseline_speed_kmh']),
        baseline_yaw_rad_s=abs(full['physical_metrics']['baseline_yaw_left_rad_s']
                              - zero['physical_metrics']['baseline_yaw_left_rad_s']),
        pulse_width_seconds=abs(width_full-width_zero))
    checks = dict(start_position=differences['start_position_m'] <=
                  spec['pair_start_position_tolerance_m'],
        baseline_speed=differences['baseline_speed_kmh'] <=
            spec['pair_baseline_speed_tolerance_kmh'],
        baseline_yaw=differences['baseline_yaw_rad_s'] <= spec['pair_baseline_yaw_tolerance_rad_s'],
        pulse_width=differences['pulse_width_seconds'] <= .01,
        unclipped=not full['pulse_clipped'] and not zero['pulse_clipped'])
    failed = []
    for path in failures:
        receipt = read(Path(path) / 'result.json')
        assert receipt['status'] == 'failed'
        failed.append(dict(path=path, result_sha256=sha256(Path(path) / 'result.json'),
                           reason=receipt['reason'], observations=receipt['observations']))
    result = dict(status='completed_partial_physical_yaw_comparison', goal_achieved=False,
        actual_complete_trials=2, actual_failed_trials=len(failed), required_complete_trials=12,
        completed_independent_observations=514, valid_input_matched_pair=all(checks.values()),
        pair_checks=checks, pair_differences=differences,
        full_pulse_seconds=width_full, zero_yaw_pulse_seconds=width_zero,
        descriptive_unqualified_iae_improvement_fraction=1-full['physical_metrics']['iae_rad']
            /zero['physical_metrics']['iae_rad'],
        full_metrics=full['physical_metrics'], zero_yaw_metrics=zero['physical_metrics'],
        failed_trials_preserved=failed, full_study_complete=False,
        physical_stabilization_verified=False, actor_qualified=False,
        navigation_verified=False, physiological_equivalence_verified=False,
        learned_longitudinal_control_verified=False, actuator_messages_sent=0,
        next_required='time-controlled actuator pulse plus validated worst-case inference warmup; '
                      'new complete comparison; do not qualify or blindly continue invalid v1')
    write_json(run / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
