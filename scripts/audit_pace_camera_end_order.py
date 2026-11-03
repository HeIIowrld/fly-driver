"""Compare failed camera copy host time to bounded CSP lifecycle clocks."""
import argparse
import json
import zipfile
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.recording)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    receipt = json.loads((root/'control/api-receipt.json').read_text(encoding='utf8'))
    bundle = json.loads((root/'vision/bundle.json').read_text(encoding='utf8'))
    archive = root/'vision/camera.zip'
    if result['status'] != 'failed' or sha256(archive) != bundle['sha256']:
        raise ValueError('Original failed camera archive required')
    with zipfile.ZipFile(archive) as z:
        failure = json.loads(z.read('shared-copy-failure.json'))
        clock = json.loads(z.read('clock-initial.json'))
    calls = {c['name']: c for c in receipt['api_calls']}
    if (clock['generation'] != receipt['before']['generation']
            or not calls['disable_ai']['returned'] or not calls['pause_after_brake_hold']['returned']):
        raise ValueError('Same-generation completed lifecycle evidence required')
    def relative(name):
        c = calls[name]
        return [failure['started_at']-(c['finished_clock']+clock['offset_upper_seconds']),
                failure['started_at']-(c['started_clock']+clock['offset_lower_seconds'])]
    run = begin_run(a.output, dict(result_sha256=sha256(root/'result.json'),
        receipt_sha256=sha256(root/'control/api-receipt.json'), archive_sha256=sha256(archive),
        program_sha256=sha256(__file__), scope='CSP clock bounds, not physical time'))
    output = dict(status='completed', original_recording_status='failed',
        failure_started_host=failure['started_at'], failure_completed_host=failure['failed_at'],
        failure_start_after_disable_ai_seconds_bounds=relative('disable_ai'),
        failure_start_after_pause_seconds_bounds=relative('pause_after_brake_hold'),
        clock_assumption=clock['unit_rate_assumption'], physical_timing_verified=False,
        causation_proven=False, fit_allowed=False)
    write_json(run/'result.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
