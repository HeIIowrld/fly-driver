"""Audit GPU event spans and host overlap, without attributing kernel utilization."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def percentiles(values):
    return None if not values else np.percentile(values, [50, 95, 100]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, audit = Path(a.recording), Path(a.audit)
    result, proof = read(root/'result.json'), read(audit/'result.json')
    manifest = read(root/'manifest.json')
    if (proof.get('status') != 'completed_mmf_immutable_shadow_audit'
            or manifest['config'].get('diagnostic_cuda_timing') is not True
            or sha256(root/'body-ticker.json') != result['ticker_sha256']
            or sha256(root/'bridge-events.json') != result['bridge_sha256']
            or sha256(audit/'ticks.json') != proof['ticks_sha256']):
        raise ValueError('Exact independently audited timed recording required')
    visual = [x['neural'] for x in read(root/'bridge-events.json')
              if x.get('neural', {}).get('visual_cuda_event_span_ms') is not None]
    ticks = read(root/'body-ticker.json')
    if len(ticks) != proof['ticks'] or not visual:
        raise ValueError('Original timed visual and ticker observations required')
    for v in visual:
        if not (0 <= v['visual_cuda_event_span_ms'] and
                v['started'] <= v['vision_finished'] <= v['finished']):
            raise ValueError('Invalid original visual timing')
    for x in ticks:
        if (x.get('body_cuda_event_span_ms') is None or x['body_cuda_event_span_ms'] < 0
                or not x['decision'] <= x['body_encode_started'] <= x['body_encode_finished']
                <= x['started'] <= x['finished']):
            raise ValueError('Invalid original body timing')
    overlap, no_overlap = [], []
    for x in ticks:
        duration = (x['finished']-x['decision'])*1000
        overlaps = any(v['started'] < x['finished'] and x['decision'] < v['vision_finished']
                       for v in visual)
        (overlap if overlaps else no_overlap).append(duration)
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_sha256': sha256(root/'result.json'), 'audit_sha256': sha256(audit/'result.json'),
        'seed': None,
        'scope': 'event span includes queue and submission gaps, not kernel occupancy'})
    summary = {'status': 'completed_mmf_shadow_gpu_timing_audit',
        'visual_events': len(visual), 'body_events': len(ticks),
        'visual_gpu_span_ms_p50_p95_max': percentiles(
            [x['visual_cuda_event_span_ms'] for x in visual]),
        'visual_host_ms_p50_p95_max': percentiles(
            [(x['vision_finished']-x['started'])*1000 for x in visual]),
        'body_gpu_span_ms_p50_p95_max': percentiles([x['body_cuda_event_span_ms'] for x in ticks]),
        'body_encode_host_ms_p50_p95_max': percentiles(
            [(x['body_encode_finished']-x['body_encode_started'])*1000 for x in ticks]),
        'body_decision_host_ms_p50_p95_max': percentiles(
            [(x['finished']-x['decision'])*1000 for x in ticks]),
        'visual_host_overlap_count': len(overlap), 'visual_host_no_overlap_count': len(no_overlap),
        'overlap_body_ms_p50_p95_max': percentiles(overlap),
        'no_overlap_body_ms_p50_p95_max': percentiles(no_overlap),
        'kernel_occupancy_or_causality_verified': False,
        'timing_instrumentation_overhead_unknown': True,
        'training_admissible': False, 'vehicle_control_sent': False}
    write_json(run/'result.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
