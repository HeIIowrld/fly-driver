"""Independently audit saved CPU timings and state-difference summaries."""

import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    paths = [base/'20261031-scipy-cpu-transfer-no-load', base/'20261031-scipy-cpu-transfer-flyvis-load']
    groups, original_hash = [], None
    for path in paths:
        result, timing, diffs = read(path/'result.json'), read(path/'contention-timing.json'), read(path/'cpu-differences.json')
        if (result['status'] != 'completed_offline_multirate_flight_replay'
                or sha256(path/'contention-timing.json') != result['timing_sha256']
                or sha256(path/'cpu-differences.json') != result['cpu_differences_sha256']
                or sha256(path/'events.json') != result['events_sha256']):
            raise ValueError('Original records and hashes required')
        original_hash = original_hash or result['events_sha256']
        if original_hash != result['events_sha256']:
            raise ValueError('Original event histories differ')
        body, visual = timing['body'], timing['visual']
        if len(body) != 387 or len(diffs) != 387 or not np.isfinite(diffs).all():
            raise ValueError('Complete original body comparisons required')
        max_difference = float(np.max(diffs))
        if max_difference != result['cpu_max_state_drive_trace_intent_abs']:
            raise ValueError('Original state comparison summary differs')
        durations = np.array([(row['finished']-row['started'])*1000 for row in body])
        transfers = np.array([(row['transfer_finished']-row['started'])*1000 for row in body if row['new_scene']])
        if len(transfers) != 53:
            raise ValueError('Every original new scene transfer required')
        overlap = np.array([any(v['started'] < row['finished'] and row['started'] < v['finished'] for v in visual) for row in body])
        if int(overlap.sum()) != result['overlap_body_count']:
            raise ValueError('Saved interval overlap differs')
        q = lambda x: np.percentile(x, [50, 95, 100]).tolist() if len(x) else None
        groups.append({'name': path.name, 'body_ms': q(durations), 'scene_transfer_ms': q(transfers),
                       'overlap_count': int(overlap.sum()), 'overlap_body_ms': q(durations[overlap]),
                       'no_overlap_body_ms': q(durations[~overlap]), 'visual_count': len(visual),
                       'max_state_drive_trace_intent_difference': max_difference})
    run = begin_run(base/'20261031-scipy-cpu-visual-contention-audit',
                    {'program_sha256': sha256(__file__), 'source_results': [sha256(x/'result.json') for x in paths],
                     'scope': 'saved host timing and GPU-vs-CPU error arrays, not live'} )
    write_json(run/'result.json', {'status': 'completed', 'groups': groups, 'live_qualified': False})
    print(json.dumps(groups))


if __name__ == '__main__':
    main()
