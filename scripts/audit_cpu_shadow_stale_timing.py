"""Decompose actual CPU-shadow stale receipts using original host timestamps."""

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_causal_copy_body_context import first_observed_packets


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root = base/'20261031-cpu-body-live-shadow'
    audit = base/'20261031-cpu-body-live-shadow-audit'
    result, independent = read(root/'result.json'), read(audit/'result.json')
    for name, key in [('body-ticker.json','ticker_sha256'), ('native-copy-windows.json','native_sha256')]:
        if sha256(root/name) != result[key]:
            raise ValueError('Original recording changed')
    if independent['status'] != 'completed_mmf_immutable_shadow_audit':
        raise ValueError('Independent source audit required')
    ticks, packets = read(root/'body-ticker.json'), first_observed_packets(read(root/'native-copy-windows.json'))
    times = [p['first_end'] for p in packets]
    failures = []
    for i, row in enumerate(ticks):
        held = row['held_body_receipt_age_at_output_seconds']
        if held is None or held <= .05:
            continue
        prev = ticks[i-1]
        latest = packets[bisect_right(times, row['decision'])-1]
        if row['body_receipt'] != times[bisect_right(times, row['body_receipt'])-1]:
            raise ValueError('Selected first receipt changed')
        failures.append({'index':i, 'selected_age_decision_ms':1000*(row['decision']-row['body_receipt']),
            'current_decision_to_output_ms':1000*(row['finished']-row['decision']),
            'previous_decision_to_output_ms':1000*(prev['finished']-prev['decision']),
            'between_decisions_ms':1000*(row['decision']-prev['decision']),
            'held_receipt_to_selected_receipt_ms':1000*(row['body_receipt']-row['held_body_receipt']),
            'held_age_at_output_ms':1000*held, 'selected_packet':row['packet'],
            'latest_available_packet_at_decision':latest['packet'],
            'native_rows_since_previous':row['native_rows_consumed']-prev['native_rows_consumed']})
    if len(failures) != 7 or len(ticks) != 250 or result['ticker_stop_reason'] != 'current_body_receipt_stale':
        raise ValueError('Unexpected failure classification')
    after = read(root/'control/native-after.json')
    if packets[-1]['packet'] != after['packet_id']:
        raise ValueError('Final captured packet and lifecycle native-after differ')
    output = begin_run(base/'20261031-cpu-shadow-stale-timing-audit',
        {'program_sha256':sha256(__file__), 'recording_result_sha256':sha256(root/'result.json'),
         'independent_audit_sha256':sha256(audit/'result.json'),
         'native_after_sha256':sha256(root/'control/native-after.json'),
         'scope':'original host receipt/order only, no physical exposure or exact stop clock'})
    write_json(output/'failures.json', failures)
    between = [(y['decision']-x['decision'])*1000 for x,y in zip(ticks,ticks[1:])]
    write_json(output/'result.json', {'status':'completed', 'held_stale_count':len(failures),
        'selected_packet_was_latest_at_decision':sum(x['selected_packet']==x['latest_available_packet_at_decision'] for x in failures),
        'selected_age_decision_ms_max':max(x['selected_age_decision_ms'] for x in failures),
        'processing_ms_min_max':[min(x['current_decision_to_output_ms'] for x in failures),max(x['current_decision_to_output_ms'] for x in failures)],
        'interdecision_ms_p50_p95_max':np.percentile(between,[50,95,100]).tolist(),
        'final_packet':packets[-1]['packet'], 'final_packet_first_receipt':packets[-1]['first_end'],
        'lifecycle_native_after_packet':after['packet_id'], 'exact_stale_stop_time_recorded':False,
        'failures_sha256':sha256(output/'failures.json'), 'vehicle_control':False})
    print(json.dumps(read(output/'result.json')))


if __name__ == '__main__':
    main()
