"""Compare frozen body-intervention replays with the same original decisions."""

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
    p.add_argument('--tilt', required=True)
    p.add_argument('--load', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    paths = [Path(a.tilt), Path(a.load)]
    results = [read(x/'result.json') for x in paths]
    manifests = [read(x/'manifest.json')['config'] for x in paths]
    for path, result, expected in zip(paths, results, ('zero_tilt', 'zero_load'), strict=True):
        if (result.get('status') != 'completed_offline_multirate_flight_replay'
                or result.get('body_intervention') != expected
                or sha256(path/'events.json') != result.get('events_sha256')
                or result.get('training_admissible') is not False):
            raise ValueError('Exact completed original intervention replay required')
    for key in ('recording_sha256', 'flyvis_sha256', 'body_audit_sha256', 'neural_audit_sha256',
                'flight_checkpoint_sha256', 'opponent_checkpoint_sha256', 'body_period_ms'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Interventions use different original input or model lineage')
    events = [read(path/'events.json') for path in paths]
    if len(events[0]) != len(events[1]) or not events[0]:
        raise ValueError('Exact same nonempty decision sequence required')
    for one, two in zip(*events, strict=True):
        if any(one[key] != two[key] for key in ('kind', 'time', 'packet', 'intent')):
            raise ValueError('Original intact decision or intent changed between interventions')
    summary = {}
    for name, data, result in zip(('tilt', 'load'), events, results, strict=True):
        delta = np.abs(np.asarray([e['intent'] for e in data]) -
                       np.asarray([e['counterfactual_intent'] for e in data]))
        if not np.isfinite(delta).all():
            raise ValueError('Finite original counterfactual output required')
        summary[name] = {'mean_abs_intent_difference': delta.mean(axis=0).tolist(),
                         'p95_abs_intent_difference': np.percentile(delta, 95, axis=0).tolist(),
                         'max_abs_intent_difference': delta.max(axis=0).tolist(),
                         'valid_original_values': result['intervention_valid_values'],
                         'original_max_abs': result['intervention_original_max_abs'],
                         'max_state_difference': result['counterfactual_state_max_abs']}
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'tilt_result_sha256': sha256(paths[0]/'result.json'),
        'load_result_sha256': sha256(paths[1]/'result.json'), 'seed': None})
    output = {'status': 'completed_body_counterfactual_audit', 'decisions': len(events[0]),
              'identical_intact_input_and_output': True, 'conditions': summary,
              'valid_zero_intervention_may_be_out_of_distribution': True,
              'physiology_or_closed_loop_improvement_verified': False,
              'training_admissible': False, 'vehicle_control_sent': False}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
