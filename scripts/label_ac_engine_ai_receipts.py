"""Audit proposal targets from future applied-control receipts; does not seal training data."""

import argparse
import base64
import hashlib
import json
import statistics
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.ac_engine_ai_labels import (
    fixed_hint_window,
    match_future_applied_controls,
    observation_segments,
    validate_incomplete_recovery,
)
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--calibration-run', required=True)
    parser.add_argument('--raw-audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-incomplete-recovery', action='store_true',
                        help='Admit a fully known bounded phase; retain failed task outcome')
    args = parser.parse_args()
    root, calibration = Path(args.recording), Path(args.calibration_run)
    audit_root = Path(args.raw_audit)
    audit = json.loads((audit_root/'result.json').read_text(encoding='utf8'))
    audit_config = json.loads((audit_root/'manifest.json').read_text(encoding='utf8'))['config']
    if (audit['status'] != 'completed' or any(audit['actor_replay_max_errors'].values())
            or audit_config['events_sha256'] != sha256(root/'events.jsonl')
            or audit_config['observation_sha256'] != sha256(root/'observations.npz')
            or audit_config['bundle_sha256'] != sha256(root/'vision/camera.zip')):
        raise ValueError('Matching successful raw pixel/native replay audit required')
    control = json.loads((root/'control/result.json').read_text(encoding='utf8'))
    receipt = json.loads((root/'control/api-receipt.json').read_text(encoding='utf8'))
    request = json.loads((root/'control/request.json').read_text(encoding='utf8'))
    partial = control['status'] != 'completed'
    if partial:
        if not args.allow_incomplete_recovery:
            raise ValueError('Failed task requires explicit phase-only admission')
        validate_incomplete_recovery(receipt, request)
    if control.get('native_motion_receipt_clock') != 'host_monotonic_QPC_time.perf_counter':
        raise ValueError('Verified motion with explicit common QPC receipt clock required')
    result = json.loads((calibration/'result.json').read_text(encoding='utf8'))
    rows = [json.loads(line)
            for line in (calibration/'samples.jsonl').read_text(encoding='utf8').splitlines()]
    medians = [statistics.median(row['native']['fields']['steerAngle'] for row in rows
               if row.get('stage') == stage and row['index'] >= 2) for stage in range(4)]
    gain = (medians[1]-medians[2])/(2*result['request_amplitude'])
    if (result['status'] != 'completed' or result['request_amplitude'] != .6
            or abs(medians[0]) > .001 or abs(medians[3]) > .001):
        raise ValueError('Completed centered stationary calibration required')
    events = [json.loads(line)
              for line in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    native = []
    for row in json.loads((root/'control/native-motion-samples.json').read_text(encoding='utf8')):
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        decoded = decode_physics_prefix(raw, row['received_monotonic_seconds'])
        if (hashlib.sha256(raw).hexdigest() != row['prefix_sha256']
                or decoded.packet_id != row['packet_id']):
            raise ValueError('Raw expert native prefix/packet hash differs')
        native.append(dict(packet_id=decoded.packet_id, fields=dict(decoded.fields),
                           received_monotonic_seconds=decoded.received_monotonic_seconds))
    if partial and not (abs(native[-1]['fields']['speedKmh']) < .1
                        and native[-1]['fields']['gas'] < .02
                        and native[-1]['fields']['brake'] > .98):
        raise ValueError('Independent native stationary brake hold required for partial admission')
    with zipfile.ZipFile(root/'vision/camera.zip') as camera:
        clock = json.loads(camera.read('clock-final.json'))
    if clock['generation'] != receipt['before']['generation']:
        raise ValueError('Camera/control clock generation differs')
    label_speed = receipt.get('label_speed_hint_kmh', receipt.get('requested_ai_speed_hint_kmh',
        request['speed_limit_kmh']))
    window = fixed_hint_window(receipt['telemetry'], label_speed,
        clock['offset_lower_seconds'], clock['offset_upper_seconds'])
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862953,
        recording_manifest_sha256=sha256(root/'manifest.json'),
        source_observation_sha256=sha256(root/'observations.npz'),
        events_sha256=sha256(root/'events.jsonl'),
        native_receipts_sha256=sha256(root/'control/native-motion-samples.json'),
        control_receipt_sha256=sha256(root/'control/api-receipt.json'),
        raw_audit_manifest_sha256=sha256(audit_root/'manifest.json'),
        calibration_samples_sha256=sha256(calibration/'samples.jsonl'),
        calibration_result_sha256=sha256(calibration/'result.json'), native_gain=gain,
        ai_window_qpc=window, label_speed_hint_kmh=label_speed,
        partial_failed_task_phase=partial, source_task_status=control['status'],
        label_kind='future40ms_applied_native_control_receipt',
        physical_command_time_verified=False))
    targets = match_future_applied_controls(events, native, native_gain=gain, ai_window=window)
    np.savez_compressed(run/'proposal-targets.npz', **targets)
    valid = targets['action_valid']
    segments = observation_segments([row['decision_monotonic_seconds'] for row in events])
    write_json(run/'result.json', dict(status='completed', observations=len(events),
        supervised_heads=valid.sum(0).tolist(), all_heads_valid_rows=int(valid.all(1).sum()),
        raw_steering_out_of_range_rows=int((np.abs(targets['raw_canonical'][:, 0]) > 1).sum()),
        observation_segments=segments, raw_values_preserved=True, clipped_labels=0,
        target_sha256=sha256(run/'proposal-targets.npz'), actor_arrays_modified=False,
        internal_ai_commands_observed=False, training_dataset_sealed=False,
        training_admissible=False, ui_actions_sent=0, actuator_messages_sent=0,
        partial_failed_task_phase=partial, source_task_success_reclassified=False,
        goal_achieved=False))


if __name__ == '__main__':
    main()
