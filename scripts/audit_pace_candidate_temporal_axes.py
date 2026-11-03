"""Audit original candidate teacher windows and raw axes without granting fit."""
import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.pedal_train_loader import (MetadataCamera, api_windows, camera_intervals,
                                              temporal_reasons)
from flybrain_drive.teacher_loss_eligibility import project_host_receipt_interval


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'raw-audit', 'authority-audit', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--role', choices=('train', 'development'), required=True)
    a = parser.parse_args()
    candidate, raw, authority = map(Path, (a.candidate, a.raw_audit, a.authority_audit))
    root = candidate/'recording'
    outer, result = read(candidate/'result.json'), read(root/'result.json')
    raw_result, raw_config = read(raw/'result.json'), read(raw/'manifest.json')['config']
    authority_result = read(authority/'result.json')
    events_path, bundle_path = root/'events.jsonl', root/'vision/camera.zip'
    if (outer['status'] != 'completed_preassigned_pace_recording' or outer['role'] != a.role
            or outer['fit_allowed'] is not False or result['training_admissible'] is not False
            or outer['recording_result_sha256'] != sha256(root/'result.json')
            or raw_result['status'] != 'completed' or any(raw_result['actor_replay_max_errors'].values())
            or raw_config['events_sha256'] != sha256(events_path)
            or raw_config['bundle_sha256'] != sha256(bundle_path)
            or authority_result['status'] != 'completed_offline_same_copy_teacher_authority_audit'
            or authority_result['source_events_sha256'] != sha256(events_path)):
        raise ValueError('Original role and matching complete source audits required')
    receipt = read(root/'control/api-receipt.json')
    generation = receipt['before']['generation']
    camera = MetadataCamera(root/'vision')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    reasons, phases, axes = Counter(), Counter(), []
    records = []
    try:
        clock = json.loads(camera.read('clock-final.json'))
        activity, windows, window_reasons = api_windows(receipt, generation)
        for i, event in enumerate(events):
            if event['index'] != i:
                raise ValueError('Full original sequence required')
            body, auth = event['native'], event['native_teacher_authority']
            data = base64.b64decode(auth['prefix_base64'], validate=True)
            if hashlib.sha256(data).hexdigest() != auth['prefix_sha256']:
                raise ValueError('Original same-copy hash differs')
            sample = decode_teacher_physics(data, body['received_monotonic_seconds'])
            if (sample.physics.packet_id != body['packet_id']
                    or sample.physics.prefix_bytes != base64.b64decode(body['prefix_base64'], validate=True)):
                raise ValueError('Same-copy body/control differs')
            frame_data = camera.read(Path(event['camera']['file']).name)
            if hashlib.sha256(frame_data).hexdigest() != event['camera']['file_sha256']:
                raise ValueError('Original frame metadata differs')
            frames = camera_intervals(event, json.loads(frame_data), generation)
            try:
                projected = project_host_receipt_interval(body['received_monotonic_seconds'],
                                                          clock, generation=generation)
            except ValueError:
                projected = None
            why, phase = temporal_reasons(sample, frames, projected, activity, windows,
                                          event['external_task_at_receipt'])
            why = list(dict.fromkeys(why+window_reasons))
            for reason in why:
                reasons[reason] += 1
            phases[phase or 'unresolved'] += 1
            f = sample.physics.fields
            row = [f['steerAngle'], f['gas'], f['brake']]
            if not np.isfinite(row).all() or not all(0 <= x <= 1 for x in row[1:]):
                raise ValueError('Original native axis values invalid')
            axes.append(row)
            records.append(dict(index=i, temporal_reasons=why, phase=phase,
                                native_axes=row, teacher_native_flag=sample.builtin_ai_controlled))
    finally:
        camera.close()
    x = np.asarray(axes)
    pedals = x[:,1:]
    subfloor = (pedals > 0) & (pedals < .02)
    coactive = (pedals > 0).all(axis=1)
    run = begin_run(a.output, dict(program_sha256=sha256(__file__), role=a.role,
        candidate_result_sha256=sha256(candidate/'result.json'), events_sha256=sha256(events_path),
        camera_sha256=sha256(bundle_path), raw_audit_sha256=sha256(raw/'result.json'),
        authority_audit_sha256=sha256(authority/'result.json'), scope=__doc__))
    write_json(run/'event-diagnostics.json', records)
    out = dict(status='completed_diagnostic_only', role=a.role, events=len(events),
        temporal_reason_counts=dict(reasons), no_temporal_reason=sum(not r['temporal_reasons'] for r in records),
        phase_counts=dict(phases), axis_min=x.min(0).tolist(), axis_max=x.max(0).tolist(),
        steering_abs_p50_p95_max=np.percentile(np.abs(x[:,0]),[50,95,100]).tolist(),
        pedal_subfloor_counts=subfloor.sum(axis=0).tolist(), pedal_coactivation_count=int(coactive.sum()),
        event_diagnostics_sha256=sha256(run/'event-diagnostics.json'),
        clock_scope='host receipt projected through saved CSP clock bounds; not physical sample time',
        moving_steering_transfer_verified=False, camera_exposure_verified=False,
        exclusive_teacher_control_verified=False, labels_created=False, fit_allowed=False,
        optimizer_updates=0)
    write_json(run/'result.json', out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
