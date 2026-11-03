"""Independently recompute timing and overlap from offline contention replays."""

import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    runs = [base/'20261031-contention-body-alone', base/'20261031-contention-body-plus-flyvis',
            base/'20261031-contention-body-alone-after']
    results = [read(p/'result.json') for p in runs]
    arrays = [read(p/'contention-timing.json') for p in runs]
    events = [read(p/'events.json') for p in runs]
    for p, result, timing in zip(runs, results, arrays, strict=True):
        if (result['status'] != 'completed_offline_multirate_flight_replay'
                or sha256(p/'contention-timing.json') != result['timing_sha256']
                or sha256(p/'events.json') != result['events_sha256']
                or len(timing['body']) != 387 or len(events[0]) != 387):
            raise ValueError('Exact original result, timing and complete event sequence required')
    if (events[0] != events[1] or events[0] != events[2] or arrays[0]['visual']
            or not arrays[1]['visual'] or arrays[2]['visual']):
        raise ValueError('Body histories must match and only one run has visual load')
    groups = {}
    for index, (name, timing) in enumerate(zip(('alone', 'loaded', 'alone_after'), arrays, strict=True)):
        body, visual = timing['body'], timing['visual']
        if any(x['finished'] < x['started'] for x in body + visual):
            raise ValueError('Invalid timing order')
        duration = np.asarray([(x['finished'] - x['started'])*1000 for x in body])
        overlap = np.asarray([any(v['started'] < x['finished'] and x['started'] < v['finished']
                                  for v in visual) for x in body])
        def stats(values):
            return None if not len(values) else np.percentile(values, [50, 95, 100]).tolist()
        groups[name] = {'count': len(body), 'all_ms_p50_p95_max': stats(duration),
                        'overlap_count': int(overlap.sum()), 'overlap_ms_p50_p95_max': stats(duration[overlap]),
                        'nonoverlap_ms_p50_p95_max': stats(duration[~overlap]),
                        'visual_count': len(visual),
                        'visual_ms_p50_p95_max': stats([(v['finished']-v['started'])*1000 for v in visual])}
        if groups[name]['overlap_count'] != results[index]['overlap_body_count']:
            raise ValueError('Overlap count disagrees with raw records')
    out = begin_run(base/'20261031-contention-independent-audit-v2',
                    {'program_sha256': sha256(__file__), 'sources': [r['timing_sha256'] for r in results],
                     'scope': 'host interval overlap and arithmetic only; workload changes CPU and GPU demand'})
    write_json(out/'result.json', {'status': 'completed', 'same_original_events': True,
                                  'groups': groups, 'gpu_occupancy_measured': False,
                                  'vehicle_control': False})
    print(json.dumps(read(out/'result.json')))


if __name__ == '__main__':
    main()
