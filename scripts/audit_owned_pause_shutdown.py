"""Audit observed pause, cooperative ticker shutdown and retained output evidence."""

import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root, audit = base/'20261031-process-body-owned-pause-live', base/'20261031-process-body-owned-pause-live-audit'
    result, source_audit = read(root/'result.json'), read(audit/'result.json')
    if (source_audit['status'] != 'completed_mmf_immutable_shadow_audit'
            or sha256(root/'body-ticker.json') != result['ticker_sha256']
            or sha256(root/'lifecycle-observations.json') != result['lifecycle_observations_sha256']):
        raise ValueError('Original recording and independently audited sources required')
    observations, ticks = read(root/'lifecycle-observations.json'), read(root/'body-ticker.json')
    pause = result['owned_pause_observation']
    if pause not in observations or not pause['fresh'] or not pause['same_generation']:
        raise ValueError('Exact original fresh owned pause observation required')
    prior = [x for x in observations if 'status' in x and x['fresh'] and x['same_generation']
             and x['finished'] < pause['finished'] and x['status']['paused'] is False
             and x['status']['expert_probe_active'] is True]
    if (not prior or pause['status']['paused'] is not True or pause['status']['ai_controlled'] is not False
            or pause['status']['expert_probe_active'] is not False
            or not pause['finished'] <= result['owned_pause_shutdown_requested'] <= result['control_exit_observed']
            or result['ticker_error'] is not None or result['ticker_stop_reason'] is not None
            or result['body_worker_exit_code'] != 0 or result['body_worker_forced_termination'] is not False
            or result['bridge_error'] is not None):
        raise ValueError('Ordered active/paused session and clean owned shutdown required')
    first_missing = ticks[0]['held_body_receipt'] is None and not ticks[0]['output_host_freshness_passed']
    passed_after_first = sum(x['output_host_freshness_passed'] for x in ticks[1:])
    run = begin_run(base/'20261031-owned-pause-shutdown-audit',
        {'program_sha256':sha256(__file__), 'recording_result_sha256':sha256(root/'result.json'),
         'raw_audit_sha256':sha256(audit/'result.json'), 'scope':'host observation and recorded shutdown order'})
    output = {'status':'completed', 'observations':len(observations),
        'observation_errors':sum('error' in x for x in observations), 'active_before_pause':True,
        'pause_observed':pause['finished'], 'shutdown_requested':result['owned_pause_shutdown_requested'],
        'request_after_observation_ms':1000*(result['owned_pause_shutdown_requested']-pause['finished']),
        'last_tick_finished':ticks[-1]['finished'], 'control_exit_observed':result['control_exit_observed'],
        'first_held_missing':first_missing, 'passed_after_first':passed_after_first,
        'after_first_count':len(ticks)-1, 'exact_pause_or_physical_exposure_verified':False,
        'worker_exit_code':result['body_worker_exit_code'], 'vehicle_control_sent':False}
    write_json(run/'result.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
