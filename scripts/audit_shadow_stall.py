"""Audit one preserved shadow run's timing without relaxing its freshness guard."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    root = Path(plan['source'])
    for name, expected in plan['inputs'].items():
        if sha256(root/name) != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    run = begin_run(args.output, dict(plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), actor=False, optimizer_updates=0))
    result = {'status': 'failed', 'actor_deployed': False, 'vehicle_control_sent': False}
    try:
        def read(name):
            return json.loads((root/name).read_text(encoding='utf8'))
        source_result = read('result.json')
        ticker = read('ticker-events.json')
        native = read('native-copy.json')
        camera = read('events.json')
        accepted = [row for row in ticker if row['status'] == 'shadow_output']
        rejected = [row for row in ticker if row['status'] == 'rejected']
        if source_result['status'] != 'stopped_on_error' or len(rejected) != 1 or not accepted:
            raise ValueError('Expected one terminal rejection and accepted records')
        bad = rejected[0]
        if bad['reason'] != 'ValueError: Current body receipt stale at output':
            raise ValueError('Unexpected failure reason')
        def summary(values):
            values = np.asarray(values, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite timing')
            return dict(count=len(values), p50=float(np.percentile(values,50)),
                p95=float(np.percentile(values,95)), p99=float(np.percentile(values,99)),
                maximum=float(values.max()))
        compute = summary([(r['finished']-r['started'])*1000 for r in accepted])
        body_age = summary([(r['finished']-r['body_receipt'])*1000 for r in accepted])
        queue = summary([(r['started']-r['decision'])*1000 for r in accepted])
        times = np.array([r['authority']['received_monotonic_seconds'] for r in native])
        if not np.all(np.diff(times) >= 0):
            raise ValueError('Native receipts out of order')
        inside = times[(times >= bad['started']) & (times <= bad['finished'])]
        if len(inside) < 2:
            raise ValueError('Insufficient native receipts in rejection window')
        overlapping = [r['capture_id'] for r in camera if r.get('context_ready') is not None
            and r.get('publication') is not None and r['context_ready'] <= bad['finished']
            and r['publication'] >= bad['started']]
        result.update(status='completed', accepted=len(accepted), rejected=len(rejected),
            accepted_compute_ms=compute, accepted_output_body_age_ms=body_age,
            accepted_queue_ms=queue, rejected_compute_ms=(bad['finished']-bad['started'])*1000,
            rejected_output_body_age_ms=(bad['finished']-bad['body_receipt'])*1000,
            native_receipts_within_rejected_window=len(inside),
            max_native_receipt_gap_ms=float(np.diff(inside).max()*1000),
            overlapping_recorded_camera_events=overlapping,
            scope='One historical stationary shadow run; host receipt/copy times, not sensor exposure or physical tick. Simultaneous gaps do not establish their cause.')
    except Exception as exc:
        result.update(error=type(exc).__name__, reason=str(exc))
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
