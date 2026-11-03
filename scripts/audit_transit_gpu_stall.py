"""Audit saved batched GPU copy journals; never alter camera timeouts or records."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    roots = [Path(value) for value in args.recording]
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': [sha256(p / 'result.json') for p in roots],
        'scope': 'host GPU call durations, not physical exposure',
        'optimizer_updates': 0, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports = []
    for root in roots:
        result = json.loads((root / 'result.json').read_text(encoding='utf8'))
        bundle = json.loads((root / 'vision/bundle.json').read_text(encoding='utf8'))
        path = root / 'vision/camera.zip'
        if sha256(path) != bundle['sha256']:
            raise ValueError('Original archived bytes must match bundle')
        with zipfile.ZipFile(path) as archive:
            journal = json.loads(archive.read('batch-copy-journal.json'))
            groups = journal['groups']
            completed = []
            for group in groups:
                if group.get('status') != 'published':
                    continue
                copies = group['copies']
                if len(copies) != 4:
                    raise ValueError('Published group must have all four original copies')
                start = copies[0]['started_at']
                flush = copies[0]['flush_finished_at']
                if not math.isfinite(start) or not math.isfinite(flush) or flush < start:
                    raise ValueError('Original host timestamps required')
                completed.append((flush-start)*1000)
            if not completed:
                raise ValueError('Published group timing required')
            a = np.asarray(completed)
            item = {'recording': str(root), 'recording_status': result['status'],
                'published_groups': len(a), 'flush_from_group_start_median_ms': float(np.median(a)),
                'flush_from_group_start_p95_ms': float(np.quantile(a, .95)),
                'flush_from_group_start_max_ms': float(a.max())}
            if 'shared-copy-failure.json' in archive.namelist():
                failure_raw = archive.read('shared-copy-failure.json')
                failure = json.loads(failure_raw)
                if (failure['error'] != 'TimeoutError: Shared four-view GPU group deadline exceeded'
                        or len(failure['stages']) != 4 or 'flush_finished_at' not in failure):
                    raise ValueError('Unexpected copy failure evidence')
                item['failure'] = {'original_failure_sha256': hashlib.sha256(
                    failure_raw).hexdigest(), 'deadline_ms': 1000*failure['timeout_seconds'],
                    'flush_from_group_start_ms': 1000*(failure['flush_finished_at'] -
                                                       failure['started_at']),
                    'first_map_started_after_ms': 1000*(failure['stages'][0]['map_started_at'] -
                                                        failure['started_at']),
                    'root_cause_not_proven': True}
            reports.append(item)
    write_json(run / 'result.json', {'status': 'completed_gpu_copy_timing_audit',
        'reports': reports, 'physical_frame_age_proven': False,
        'timeouts_modified': False, 'actor_deployed': False})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
