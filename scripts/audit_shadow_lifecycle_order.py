"""Audit observed CSP pause versus shadow stop without inventing event times."""

import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root, audit = base/'20261031-process-body-lifecycle-observed', base/'20261031-process-body-lifecycle-observed-audit'
    result, proof = read(root/'result.json'), read(audit/'result.json')
    if (proof['status'] != 'completed_mmf_immutable_shadow_audit'
            or sha256(root/'lifecycle-observations.json') != result['lifecycle_observations_sha256']
            or sha256(root/'body-ticker.json') != result['ticker_sha256']):
        raise ValueError('Original observations and independent input audit required')
    observations = read(root/'lifecycle-observations.json')
    ticks, after = read(root/'body-ticker.json'), read(root/'control/native-after.json')
    valid = [x for x in observations if 'status' in x and x['fresh'] and x['same_generation']]
    if any(x['started'] > x['finished'] for x in observations):
        raise ValueError('Original observation order invalid')
    active = [x for x in valid if x['status']['paused'] is False and x['status']['expert_probe_active'] is True]
    if not active:
        raise ValueError('A fresh active owned session must be observed first')
    first_active = active[0]['finished']
    pauses = [x for x in valid if x['finished'] > first_active and x['status']['paused'] is True
              and x['status']['ai_controlled'] is False and x['status']['expert_probe_active'] is False]
    if not pauses:
        raise ValueError('No subsequent fresh paused session observed')
    first_pause = pauses[0]
    stop = result['ticker_stop_details']
    if result['ticker_stop_reason'] != 'current_body_receipt_stale' or stop['packet'] != after['packet_id']:
        raise ValueError('Original stale stop and terminal native packet required')
    gap = (stop['time']-first_pause['finished'])*1000
    rows = {'status':'completed', 'observations':len(observations),
            'observation_errors':sum('error' in x for x in observations),
            'fresh_same_generation_observations':len(valid),
            'first_active_observed':first_active, 'first_post_active_pause_observation':first_pause,
            'stale_stop_time':stop['time'], 'pause_observed_before_stop':gap >= 0,
            'pause_observation_to_stop_ms':gap, 'stop_packet':stop['packet'],
            'terminal_native_packet':after['packet_id'], 'last_tick_finished':ticks[-1]['finished'],
            'exact_pause_instant_or_sensor_exposure_verified':False,
            'original_run_reclassified':False, 'vehicle_control_sent':False}
    run = begin_run(base/'20261031-process-lifecycle-order-audit',
        {'program_sha256':sha256(__file__), 'recording_result_sha256':sha256(root/'result.json'),
         'independent_audit_sha256':sha256(audit/'result.json'), 'native_after_sha256':sha256(root/'control/native-after.json'),
         'scope':'fresh published-state observations in host time, no exact pause or sensor time'})
    write_json(run/'result.json', rows)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
