"""Audit actual neural shadow receipts, image hashes, output ages and observed overlap."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--body-audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, audit = Path(a.recording), Path(a.body_audit)
    read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
    sr, br = read(root/'result.json'), read(audit/'result.json')
    bm = read(audit/'manifest.json')['config']
    if (sr['status'] != 'completed' or sr['training_admissible'] is not False
            or sr['shadow_sha256'] != sha256(root/'online-body-shadow.json')
            or sr['metadata_sha256'] != sha256(root/'diagnostic-frames.json')
            or br['status'] != 'completed_online_body_shadow_audit'
            or bm['recording_sha256'] != sha256(root/'result.json')):
        raise ValueError('Exact completed live recording and original body audit required')
    events, rows = read(root/'online-body-shadow.json'), read(root/'diagnostic-frames.json')
    clock = read(root/'clock-before.json')
    if len(events) != len(rows) or len(events) != sr['frames']:
        raise ValueError('One original neural event per camera group required')
    parsed, previous_flight_receipt = [], None
    names = ('front', 'left', 'right', 'rear')
    for i, (event, row) in enumerate(zip(events, rows, strict=True)):
        path = Path(row['source_file'])
        if (row['index'] != i or event['index'] != i or sha256(path) != row['sha256']
                or event['metadata_sha256'] != row['sha256'] or 'neural_error' in event):
            raise ValueError('Exact unfailed neural and camera metadata required')
        frame, n = read(path), event['neural']
        views = {v['view_id']: v for v in frame['views']}
        if set(views) != set(names) or n['vehicle_control'] is not False:
            raise ValueError('Original four images and no vehicle control required')
        times = ([event['camera_receipt'], event['visual_snapshot'], n['started'],
                  n['vision_finished'], event['decision'], event['context_ready'], n['finished']]
                 if 'visual_snapshot' in event else
                 [event['camera_receipt'], event['decision'], event['context_ready'],
                  n['started'], n['finished']])
        if any(b < a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError('Causal host shadow pipeline times required')
        for j, name in enumerate(names):
            if sha256(path.parent/views[name]['file']) != n['source_images_sha256'][j]:
                raise ValueError('Image differs from its original neural-read hash')
        ages = [n['finished']-bound for bound in event['render_start_lower']]
        np.testing.assert_allclose(ages, n['view_render_age_upper_at_output_seconds'], rtol=0, atol=1e-9)
        if abs(n['finished']-event['camera_receipt']-n['camera_receipt_to_output_seconds']) > 1e-9:
            raise ValueError('Camera receipt/output duration differs')
        if n['intent'] is not None:
            body = event['decision_body']
            if (n['status'] != 'intent' or not np.isfinite(n['intent']).all()
                    or abs(n['finished']-body['receipt']-n['body_receipt_age_at_output_seconds']) > 1e-9):
                raise ValueError('Finite intent and exact body receipt age required')
            if 'previous_held_body_receipt' in n:
                if n['previous_held_body_receipt'] != previous_flight_receipt:
                    raise ValueError('Output must identify the body receipt held during its interval')
                if previous_flight_receipt is not None and abs(
                        n['finished']-previous_flight_receipt-
                        n['previous_held_body_receipt_age_at_output_seconds']) > 1e-9:
                    raise ValueError('Previous held body age differs')
                previous_flight_receipt = body['receipt']
        if 'output_host_freshness_passed' in n:
            expected_fresh = (n['intent'] is not None
                and 0 <= n['body_receipt_age_at_output_seconds'] <= .05
                and n['previous_held_body_receipt_age_at_output_seconds'] is not None
                and 0 <= n['previous_held_body_receipt_age_at_output_seconds'] <= .05
                and max(ages) <= .15)
            if n['output_host_freshness_passed'] != expected_fresh:
                raise ValueError('Output-time host freshness declaration differs')
        save_upper = max(views[name]['save_finished_clock'] for name in names)+clock['offset_upper_seconds']
        parsed.append({'save_upper': save_upper, 'n': n, 'e': event})
    intents = [row for row in parsed if row['n']['intent'] is not None]
    if not intents:
        raise ValueError('No live neural flight outputs to audit')
    ms = lambda values: np.percentile(np.asarray(values)*1000, [50, 95, 100]).tolist()
    body = [row['n']['body_receipt_age_at_output_seconds'] for row in intents]
    held_body = [row['n']['previous_held_body_receipt_age_at_output_seconds'] for row in intents
                 if row['n'].get('previous_held_body_receipt_age_at_output_seconds') is not None]
    view_ages = [max(row['n']['view_render_age_upper_at_output_seconds']) for row in intents]
    overlap = sum(parsed[i]['n']['finished'] > parsed[i+1]['save_upper']
                  for i in range(len(parsed)-1))
    ready = [row['e']['context_ready']-row['e']['camera_receipt'] for row in parsed]
    processing = [row['n']['processing_seconds'] for row in parsed]
    output_age = [row['n']['camera_receipt_to_output_seconds'] for row in parsed]
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': sha256(root/'result.json'),
        'body_audit_result_sha256': sha256(audit/'result.json'),
        'clock_before_sha256_at_audit': sha256(root/'clock-before.json'),
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    out = {'status': 'completed_live_neural_shadow_audit', 'groups': len(events),
        'intent_groups': len(intents), 'missing_required_body_groups': sum(
            x['n']['status'] == 'missing_required_body' for x in parsed),
        'flight_stopped_after_gap_groups': sum(x['n']['status'] == 'flight_stopped_after_body_gap'
                                                for x in parsed),
        'context_ready_after_camera_ms_p50_p95_max': ms(ready),
        'neural_process_ms_p50_p95_max': ms(processing),
        'camera_receipt_to_output_ms_p50_p95_max': ms(output_age),
        'intent_body_receipt_age_ms_p50_p95_max': ms(body),
        'intent_body_receipt_age_over_50ms': sum(x > .05 for x in body),
        'previous_held_body_count': len(held_body),
        'previous_held_body_age_ms_p50_p95_max': ms(held_body) if held_body else None,
        'previous_held_body_age_over_50ms': sum(x > .05 for x in held_body),
        'output_host_freshness_passed': sum(x['n'].get('output_host_freshness_passed') is True
                                             for x in intents),
        'intent_max_view_render_age_upper_ms_p50_p95_max': ms(view_ages),
        'intent_view_render_age_upper_over_150ms': sum(x > .15 for x in view_ages),
        'output_after_next_group_save_upper_count': overlap,
        'overlap_is_not_proven_metadata_queue': True,
        'age_is_host_receipt_not_physical_sample_time': True,
        'clock_before_hash_originally_sealed': False, 'training_admissible': False,
        'physical_or_exposure_time_verified': False, 'vehicle_control_sent': False}
    write_json(run/'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
