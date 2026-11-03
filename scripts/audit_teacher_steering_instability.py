"""Read-only comparison of recorded teacher steering and host receipt latency.

Region .76..84 is the Monza straight after Ascari. Descriptive comparison only:
different trajectories, timing and samples do not identify an isolated cause.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def audit(path):
    result = json.loads((path/'result.json').read_text(encoding='utf8'))
    digest = sha256(path/'events.jsonl')
    if digest != result['events_sha256']:
        raise ValueError('Recorded event archive differs')
    rows, pair, feedback = [], None, None
    for line in (path/'events.jsonl').open(encoding='utf8'):
        event = json.loads(line)
        if event['kind'] == 'raw_pair':
            pair = event
        elif event['kind'] == 'feedback':
            feedback = event
        elif event['kind'] == 'teacher_request' and pair:
            sdk = pair['sdk']
            payload = base64.b64decode(sdk['wire_base64'], validate=True)
            if hashlib.sha256(payload).hexdigest() != sdk['wire_sha256']:
                raise ValueError('SDK payload differs')
            packet = json.loads(payload)
            rows.append([packet['privileged_teacher']['raw']['NormalizedSplinePosition'],
                (event['host_time']-sdk['received_monotonic_seconds'])*1000,
                (pair['native']['received_monotonic_seconds']-sdk['received_monotonic_seconds'])*1000,
                event['limited'][0], event['observation'][9]])
    data = np.asarray(rows)
    if not data.size:
        raise ValueError('Teacher requests required')
    straight = data[(data[:, 0] > .76) & (data[:, 0] < .84)]
    return dict(recording=str(path), events_sha256=digest, status=result['status'],
        termination_reason=result.get('reason'), teacher_requests=len(rows),
        sdk_receipt_age_at_request_p50_p95_max_ms=np.percentile(data[:, 1], [50,95,100]).tolist(),
        sdk_native_receipt_skew_p50_p95_max_ms=np.percentile(data[:, 2], [50,95,100]).tolist(),
        straight_region_progress=[.76,.84], straight_requests=len(straight),
        straight_abs_steering_p95_max=np.percentile(abs(straight[:, 3]), [95,100]).tolist(),
        straight_abs_normalized_lateral_velocity_max=float(abs(straight[:, 4]).max()),
        straight_receipt_age_p50_p95_max_ms=np.percentile(straight[:, 1], [50,95,100]).tolist(),
        last_feedback=feedback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = begin_run(args.output, dict(arguments=vars(args), seed=780102,
        actuator_messages_sent=0, ui_actions_sent=0, teacher_only=True))
    comparisons = [audit(Path(path)) for path in args.recordings]
    write_json(run/'result.json', dict(status='completed', recordings=comparisons,
        interpretation='Descriptive association, not isolated causation or repeated-run reliability',
        actuator_messages_sent=0, student_optimizer_updates=0))
    print(json.dumps(comparisons))


if __name__ == '__main__':
    main()
