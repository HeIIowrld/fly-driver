"""Summarize warm and first-flight timing from an existing offline single-stream benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.benchmark)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    if (result['status'] != 'completed_integrated_offline_single_stream_benchmark'
            or result['training_admissible'] is not False
            or sha256(root/'timings.npz') != result['arrays_sha256']):
        raise ValueError('Exact original completed offline benchmark required')
    with np.load(root/'timings.npz', allow_pickle=False) as data:
        total, stages, queue, clocks = (data[key].copy() for key in
            ('total_ms', 'stage_ms', 'idealized_queue_sojourn_ms', 'decision_clock'))
    start, count = result['flight_segment_start'], result['flight_groups']
    stop = start+count
    if (stages.shape != (len(total), 5) or len(total) != len(queue) or len(total) != len(clocks)
            or count < 2 or stop > len(total) or not np.isfinite(stages).all()
            or not np.all(np.diff(clocks) > 0)):
        raise ValueError('Complete finite original ordered measurements required')
    percentile = lambda x: np.percentile(x, [50, 95, 100]).tolist()
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'benchmark_result_sha256': sha256(root/'result.json'),
        'original_timings_sha256': result['arrays_sha256'], 'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    out = {'status': 'completed_offline_timing_audit', 'flight_groups': count,
        'first_flight_group': start, 'first_flight_total_ms': float(total[start]),
        'first_flight_stage_ms': float(stages[start, 4]),
        'flight_stage_ms_p50_p95_max': percentile(stages[start:stop, 4]),
        'after_first_flight_total_ms_p50_p95_max': percentile(total[start+1:stop]),
        'recorded_selection_interval_ms_p50_p95_min': np.percentile(np.diff(clocks)*1000,
                                                                      [50, 95, 0]).tolist(),
        'idealized_queue_peak_group': int(np.argmax(queue)),
        'idealized_queue_peak_sojourn_ms': float(np.max(queue)),
        'idealized_queue_last_sojourn_ms': float(queue[-1]),
        'queue_is_not_a_live_execution': True, 'body_staleness_under_queue_not_evaluated': True,
        'training_admissible': False, 'actor_deployed': False}
    write_json(run/'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
