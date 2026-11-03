"""Seal audited moving-handoff expert targets without cutting the student's recurrent prefix."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.training_source_guard import require_training_source
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_engine_ai_labels import fixed_hint_window
from flybrain_drive.ac_expert_handoff import handoff_observation_records
from flybrain_drive.ac_expert_handoff_labels import handoff_applied_targets
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_native_receipts import verified_handoff_native_receipts
from flybrain_drive.ac_student_pilot import has_applied_receipt_task
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes
from flybrain_drive.multiview_sensory import MultiViewPacker


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--prefix-audit', required=True)
    parser.add_argument('--raw-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, prefix, audit = map(Path, (args.trial, args.prefix_audit, args.raw_audit))
    config, outcome = read(root/'manifest.json')['config'], read(root/'result.json')
    full, full_config = read(audit/'result.json'), read(audit/'manifest.json')['config']
    before = read(prefix/'result.json')
    require_training_source(recording=outcome, raw_audit=full, label_audit=before)
    if (outcome['status'] != 'expert_handoff_completed' or not outcome['post_stop_verified']
            or outcome['final_vjoy_status'] != 1 or before['status'] != 'completed'
            or any(before['replay_max_absolute_error']) or full['status'] != 'completed'
            or not full['actor_replay_max_errors']
            or any(full['actor_replay_max_errors'].values())
            or full['prefix_and_waiting_neural_replay_max_error'] != 0
            or not full['same_retina_history_across_handoff']
            or not full['native_stop_verified'] or full['intervals_above80ms'] != 0
            or full_config['events_sha256'] != sha256(root/'events.jsonl')
            or full_config['observations_sha256'] != sha256(root/'handoff-observations.npz')
            or full_config['prefix_audit_manifest_sha256'] != sha256(prefix/'manifest.json')
            or full_config['expert_receipt_sha256']
                != sha256(root/'expert-handoff/api-receipt.json')
            or read(prefix/'manifest.json')['config']['trial_events_sha256']
                != sha256(root/'events.jsonl')):
        raise ValueError('Matching exact full-history and prefix audits with no gaps required')
    contract = config['student_teacher_task_contract']
    request = read(root/'expert-handoff/request.json')
    receipt = read(root/'expert-handoff/api-receipt.json')
    if (not has_applied_receipt_task(contract, 15, .6)
            or (request['command'], request['hold_seconds']) not in (
                ('expert-moving-handoff3', 3), ('expert-moving-handoff30', 30))
            or request['speed_limit_kmh'] != 15 or request['ai_steer_multiplier'] != .9
            or receipt['status'] != 'api_returned_true' or not receipt['moving_handoff_passed']
            or request['id'] != receipt['id']):
        raise ValueError('Exact bounded15kmh future-applied receipt task required')
    calibration_path = Path(config['arguments']['calibration_result']).parent/'samples.jsonl'
    if (sha256(calibration_path) != before['native_steering_calibration_source_sha256']
            or read(root/'calibration-input.json')['request_amplitude'] != .6):
        raise ValueError('Unchanged independently verified stationary steering codec required')
    gain = before['native_steering_calibration_gain']
    events = [json.loads(line) for line in
              (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    with load_array(root/'handoff-observations.npz', allow_pickle=False) as data:
        original = {key: data[key] for key in data.files}
    if original['teacher_phase'].dtype != np.bool_:
        raise ValueError('Original boolean coordinator role array required')
    records, pairs = handoff_observation_records(events, original['teacher_phase'])
    actor = {key: value for key, value in original.items()
             if key.startswith('obs_') or key == 'observation_time'}
    if (set(key[4:] for key in actor if key.startswith('obs_')) != MultiViewPacker.fields
            or any(len(value) != len(records) for value in original.values())
            or not actor['obs_view_valid'].all()
            or not np.array_equal(actor['observation_time'],
                [row['decision_monotonic_seconds'] for row in records])):
        raise ValueError('Unmodified complete four-view actor history required')
    native = []
    for row in pairs.values():
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
        if hashlib.sha256(raw).hexdigest() != row['prefix_sha256'] or (
                sample.packet_id != row['packet_id']):
            raise ValueError('Original raw native hash/packet differs')
        native.append(dict(packet_id=sample.packet_id, fields=dict(sample.fields),
                           received_monotonic_seconds=sample.received_monotonic_seconds))
    independent = None
    if config['arguments'].get('handoff_native_receipts'):
        native, independent = verified_handoff_native_receipts(root, events)
        if independent != full.get('independent_native_receipt_capture'):
            raise ValueError('Independent native receipts differ from full raw audit')
    with CameraArchive(root/'vision') as camera:
        if camera.bundle_sha256 != outcome['camera_archive_receipt']['bundle_sha256']:
            raise ValueError('Original camera bundle receipt differs')
        initial = camera.json('clock-initial.json')
    clock = CSPClockBounds(initial['generation'],
        maximum_width=initial['maximum_width_seconds'],
        host_quantization=initial.get('host_clock_quantization_seconds', 0.))
    for ping in initial['exchanges']:
        clock.observe(ping['sent'], ping['received'], ping['producer_clock'], initial['generation'])
    if clock.record() != initial or clock.generation != receipt['before']['generation']:
        raise ValueError('Original same-generation causal clock bounds required')
    observed_window = fixed_hint_window(receipt['telemetry'], 15, clock.low, clock.high)
    released_at = next(e['released_at'] for e in events if e['kind'] == 'handoff_vjoy_released')
    # vJoy close submits brake1; exclude its transient from expert control targets.
    window = (max(observed_window[0], released_at+.3), observed_window[1])
    targets = handoff_applied_targets(records, native, native_gain=gain, ai_window=window)
    roles = original['teacher_phase']
    if targets['action_valid'][~roles].any():
        raise ValueError('Student/waiting observations cannot acquire expert targets')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862991,
        trial_manifest_sha256=sha256(root/'manifest.json'),
        events_sha256=sha256(root/'events.jsonl'),
        source_observations_sha256=sha256(root/'handoff-observations.npz'),
        raw_audit_manifest_sha256=sha256(audit/'manifest.json'),
        raw_audit_result_sha256=sha256(audit/'result.json'),
        prefix_audit_manifest_sha256=sha256(prefix/'manifest.json'),
        prefix_audit_result_sha256=sha256(prefix/'result.json'),
        expert_receipt_sha256=sha256(root/'expert-handoff/api-receipt.json'),
        expert_request_sha256=sha256(root/'expert-handoff/request.json'),
        calibration_samples_sha256=sha256(calibration_path), native_gain=gain,
        ai_window_qpc=window, camera_bundle_sha256=camera.bundle_sha256,
        observed_ai_window_qpc=observed_window, vjoy_release_settle_seconds=.3,
        independent_native_receipt_capture=independent))
    np.savez_compressed(run/'proposal-targets.npz', **targets)
    episode = run/'episode-00.npz'
    np.savez_compressed(episode, **actor, action=targets['action'],
                        action_valid=targets['action_valid'])
    with load_array(episode, allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], value) for key, value in actor.items()):
            raise ValueError('Sealing changed original actor inputs or clocks')
    counts = targets['action_valid'].sum(0).tolist()
    calibration = CSPCameraRetina(config['pixel_configuration']).calibration
    write_json(run/'dataset.json', dict(schema='flybrain.multiview.teacher.v1',
        action_profile='normalized_ac_absolute', sensory_calibration=calibration,
        actor_fields=sorted(MultiViewPacker.fields), decision_dt_seconds=.04,
        brain_ticks_per_decision=4, decisions=len(records),
        episodes=[dict(file=episode.name, sha256=sha256(episode), steps=len(records),
            episode_id='moving-handoff-complete-history', original_trial=args.trial,
            boundary='original episode start only; no reset at AI transfer',
            original_observation_start=0, original_observation_stop=len(records),
            source_actor_observation_sha256=sha256(root/'handoff-observations.npz'))],
        teacher_kind='engine_ai acknowledged suffix only; student prefix unsupervised',
        teacher_label_contract=contract, supervised_action_counts=counts,
        expert_labels_available=True, labels_are_executed_requests=False,
        labels_are_executed_applied_controls=True, native_AC_training_data=True,
        training_admissible=True, experimental_offline_replay_only=True,
        actual_observation_clocks_preserved=True, recurrent_prefix_preserved=True,
        student_and_waiting_label_counts=[0, 0, 0], internal_ai_commands_observed=False,
        full_lap_verified=False, autonomous_qualification=False,
        limitations='Future applied-control surrogate; physical command latency unverified. '
                    'No student imitation labels and no race-optimal expert claim.'))
    load_episodes(run, calibration)
    write_json(run/'result.json', dict(status='completed', episodes=1, observations=len(records),
        student_prefix_observations=sum(not r['expert'] and not r['bridge'] for r in records),
        waiting_observations=sum(r['bridge'] for r in records),
        expert_suffix_observations=int(roles.sum()), supervised_heads=counts,
        student_and_waiting_supervised_heads=[0, 0, 0],
        maximum_observation_interval_seconds=float(np.diff(actor['observation_time']).max()),
        actor_arrays_equal_source=True, recurrent_prefix_preserved=True,
        handoff_state_reset=False, clipped_labels=0,
        positive_brake_labels=int(((targets['action'][:, 2] > .05)
                                   & targets['action_valid'][:, 2]).sum()),
        training_admissible=True, experimental_surrogate_only=True,
        target_sha256=sha256(run/'proposal-targets.npz'),
        dataset_manifest_sha256=sha256(run/'dataset.json'),
        ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
