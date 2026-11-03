"""Predeclared numerical refinement checks on every contact-current hypothesis."""

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
    parser.add_argument('--refined', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    refined = Path(args.refined)
    plan = read(refined/'manifest.json')['config']['plan']
    limits = plan['numerical_refinement']
    coarse = Path(limits['coarse_result'])
    before, after = read(coarse/'result.json'), read(refined/'result.json')
    assert before['status'] == after['status'] == 'completed_fast_haltere_hypothesis_probe'
    assert len(before['cases']) == len(after['cases']) == 120
    run = begin_run(args.output, dict(arguments=vars(args), limits=limits,
        coarse_sha256=sha256(coarse/'result.json'), refined_sha256=sha256(refined/'result.json')))
    rows = []
    for a, b in zip(before['cases'], after['cases'], strict=True):
        for key in ('hypothesis', 'frequency_hz', 'amplitude_rad_s', 'cadence_ticks',
                    'coupling', 'fast_tau_seconds'):
            assert a[key] == b[key]
        gain = abs(a['gain_per_ideal_yaw']/b['gain_per_ideal_yaw']-1)
        phase = abs((a['opposing_intent_response']['phase_degrees']
                     -b['opposing_intent_response']['phase_degrees']+180) % 360-180)
        neutral = float(np.max(np.abs(np.array(a['neutral_intent'])-b['neutral_intent'])))
        rows.append(dict(hypothesis=a['hypothesis'], frequency_hz=a['frequency_hz'],
            amplitude_rad_s=a['amplitude_rad_s'], cadence_ticks=a['cadence_ticks'],
            relative_gain_difference=gain, phase_difference_degrees=phase,
            neutral_intent_difference=neutral,
            passed=(gain <= limits['maximum_relative_gain_difference']
                    and phase <= limits['maximum_phase_difference_degrees']
                    and neutral <= limits['maximum_neutral_intent_difference'])))
    write_json(run/'result.json', dict(status='completed_numerical_refinement_comparison',
        goal_achieved=False, actuator_messages_sent=0, optimizer_updates=0,
        all_cases_passed=all(row['passed'] for row in rows), cases=rows,
        maximum_relative_gain_difference=max(row['relative_gain_difference'] for row in rows),
        maximum_phase_difference_degrees=max(row['phase_difference_degrees'] for row in rows),
        maximum_neutral_intent_difference=max(row['neutral_intent_difference'] for row in rows),
        physical_stability_repaired=False, physiology_verified=False))


if __name__ == '__main__':
    main()
