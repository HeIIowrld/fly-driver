"""Summarize raw steering magnitude without claiming actuator calibration."""
import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.audit)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    rows_path = root/'event-diagnostics.json'
    if result['status'] != 'completed_diagnostic_only' or sha256(rows_path) != result['event_diagnostics_sha256']:
        raise ValueError('Original diagnostic event summary required')
    rows = json.loads(rows_path.read_text(encoding='utf8'))
    if len(rows) != result['events']:
        raise ValueError('Complete original event count required')
    run = begin_run(a.output, dict(program_sha256=sha256(__file__),
        audit_sha256=sha256(root/'result.json'), rows_sha256=sha256(rows_path), scope=__doc__))
    out = dict(status='completed_diagnostic_only', role=result['role'], events=len(rows),
        raw_native_abs_steer_above_01=sum(abs(r['native_axes'][0]) > .1 for r in rows),
        raw_native_abs_steer_above_05=sum(abs(r['native_axes'][0]) > .5 for r in rows),
        raw_native_abs_steer_above_08=sum(abs(r['native_axes'][0]) > .8 for r in rows),
        thresholds_are_native_values_not_calibrated_actuator_limits=True,
        labels_created=False, fit_allowed=False)
    write_json(run/'result.json',out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
