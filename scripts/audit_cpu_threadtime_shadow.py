"""Audit CPU-thread timer quantization and stale stop on retained live receipts."""

import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root, audit = base/'20261031-cpu-body-threadtime-live', base/'20261031-cpu-body-threadtime-live-audit'
    result, proof = read(root/'result.json'), read(audit/'result.json')
    if (proof['status'] != 'completed_mmf_immutable_shadow_audit'
            or sha256(root/'body-ticker.json') != result['ticker_sha256']
            or sha256(root/'native-copy-windows.json') != result['native_sha256']):
        raise ValueError('Original independently audited recording required')
    ticks = read(root/'body-ticker.json')
    if len(ticks) != proof['ticks'] or any(x['body_thread_cpu_seconds'] is None for x in ticks):
        raise ValueError('All original CPU clock samples required')
    wall = np.array([(x['finished']-x['started'])*1000 for x in ticks])
    cpu = np.array([x['body_thread_cpu_seconds']*1000 for x in ticks])
    if not np.isfinite(cpu).all() or np.any(cpu < 0):
        raise ValueError('Nonnegative original thread CPU times required')
    rounded, counts = np.unique(np.round(cpu, 3), return_counts=True)
    stop = result['ticker_stop_details']
    after = read(root/'control/native-after.json')
    if (result['ticker_stop_reason'] != 'current_body_receipt_stale'
            or stop['packet'] != after['packet_id'] or stop['time'] < ticks[-1]['finished']
            or abs(stop['age_seconds']-(stop['time']-stop['first_receipt'])) > 1e-9):
        raise ValueError('Original terminal packet or timing differs')
    run = begin_run(base/'20261031-cpu-threadtime-independent-audit',
        {'program_sha256':sha256(__file__), 'recording_result_sha256':sha256(root/'result.json'),
         'independent_audit_sha256':sha256(audit/'result.json'),
         'native_after_sha256':sha256(root/'control/native-after.json'),
         'scope':'original host timestamps and OS thread timer, no actual sensor exposure time'})
    result = {'status':'completed', 'ticks':len(ticks), 'freshness_passed':proof['host_freshness_passed'],
        'wall_ms_p50_p95_max':np.percentile(wall,[50,95,100]).tolist(),
        'thread_cpu_ms_p50_p95_max':np.percentile(cpu,[50,95,100]).tolist(),
        'thread_cpu_ms_unique_rounded':rounded.tolist(), 'thread_cpu_counts':counts.tolist(),
        'thread_cpu_exceeds_wall_count':int((cpu>wall).sum()),
        'per_decision_cpu_wait_separation_usable':False, 'stop_details':stop,
        'native_after_packet':after['packet_id'], 'native_after_host_receipt':after['received_monotonic_seconds'],
        'physical_sample_time_verified':False, 'pause_exact_host_time_verified':False,
        'vehicle_control_sent':False}
    write_json(run/'result.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
