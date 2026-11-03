"""Recompute cross-process timing and state evidence from saved replays."""

import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    roots = [base/'20261031-cpu-process-visual-no-load', base/'20261031-cpu-process-visual-load']
    summaries, event_hash = [], None
    for root in roots:
        result, timing, diffs = read(root/'result.json'), read(root/'contention-timing.json'), read(root/'cpu-differences.json')
        if (result['status'] != 'completed_offline_multirate_flight_replay' or result['cpu_worker_exit_code'] != 0
                or sha256(root/'events.json') != result['events_sha256']
                or sha256(root/'contention-timing.json') != result['contention_timing_sha256']
                or sha256(root/'cpu-differences.json') != result['cpu_differences_sha256']):
            raise ValueError('Original completed replay, worker and hashes required')
        event_hash = event_hash or result['events_sha256']
        if result['events_sha256'] != event_hash or len(timing['body']) != 387 or len(diffs) != 387:
            raise ValueError('Same complete original event history required')
        if not np.isfinite(diffs).all() or float(np.max(diffs)) != result['cpu_max_state_drive_trace_intent_abs']:
            raise ValueError('All original state differences required')
        body, visual = timing['body'], timing['visual']
        for x in body:
            if not x['started'] <= x['transfer_finished'] <= x['worker_started'] <= x['worker_finished'] <= x['finished']:
                raise ValueError('Original cross-process host ordering required')
        wall = np.array([(x['finished']-x['started'])*1000 for x in body])
        child = np.array([(x['worker_finished']-x['worker_started'])*1000 for x in body])
        transfer = np.array([(x['transfer_finished']-x['started'])*1000 for x in body if x['new_scene']])
        overhead = wall-child
        overlap = np.array([any(v['started']<x['finished'] and x['started']<v['finished'] for v in visual) for x in body])
        if len(transfer) != 53 or int(overlap.sum()) != result['overlap_body_count']:
            raise ValueError('Original scene count and overlap required')
        q = lambda x: np.percentile(x, [50,95,100]).tolist() if len(x) else None
        summaries.append({'name':root.name, 'samples':len(body), 'wall_ms':q(wall), 'worker_ms':q(child),
                          'wall_minus_worker_ms':q(overhead), 'new_scene_copy_ms':q(transfer),
                          'overlap_count':int(overlap.sum()), 'overlap_wall_ms':q(wall[overlap]),
                          'visual_steps':len(visual), 'max_state_difference':float(np.max(diffs)),
                          'worker_pid':result['cpu_worker_pid']})
    run = begin_run(base/'20261031-cpu-process-visual-audit', {'program_sha256':sha256(__file__),
        'source_results':[sha256(p/'result.json') for p in roots], 'scope':'saved host-clock/IPC timings, not live'})
    write_json(run/'result.json', {'status':'completed', 'summaries':summaries,
                                   'live_freshness_verified':False, 'vehicle_control_sent':False})
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
