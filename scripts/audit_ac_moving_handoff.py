"""Replay the entire actual student-prefix/AI-suffix sensory history, without reset or labels."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_engine_ai_labels import fixed_hint_window
from flybrain_drive.ac_expert_handoff import (
    effective_motion_after_release,
    handoff_observation_records,
    validate_curve_handoff_evidence,
    validate_handoff_preset,
)
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_native_receipts import verified_handoff_native_receipts
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--prefix-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.trial)
    config = read(root/'manifest.json')['config']
    result = read(root/'result.json')
    prefix = Path(args.prefix_audit)
    prefix_result = read(prefix/'result.json')
    if (result['status'] != 'expert_handoff_completed' or not result['post_stop_verified']
            or result['final_vjoy_status'] != 1 or not config['arguments']['expert_handoff']
            or prefix_result['status'] != 'completed'
            or any(prefix_result['replay_max_absolute_error'])
            or read(prefix/'manifest.json')['config']['trial_events_sha256']
                != sha256(root/'events.jsonl')):
        raise ValueError('Successful handoff and matching exact prefix audit required')
    receipt = read(root/'expert-handoff/api-receipt.json')
    request = read(root/'expert-handoff/request.json')
    arguments = config['arguments']
    validate_handoff_preset(arguments.get('handoff_context', 'launch'),
        arguments['decisions'], request['hold_seconds'],
        arguments.get('handoff_native_receipts', False), arguments['trial_profile'],
        arguments['camera_backend'], arguments['require_background'])
    validate_curve_handoff_evidence(arguments, request, prefix_result)
    activation = read(root/'expert-handoff/activation.json')
    before, after = receipt['before'], receipt['after']
    expected_calls = ['enable_ai', 'set_ai_steer_multiplier', 'set_ai_speed_limit', 'disable_ai',
                      'pause_after_brake_hold', 'restore_default_ai_speed_limit',
                      'restore_default_ai_steer_multiplier']
    duration = request['hold_seconds']
    if ((request['command'], duration) not in (
                ('expert-moving-handoff3', 3), ('expert-moving-handoff30', 30))
            or request['speed_limit_kmh'] != 15 or request['ai_steer_multiplier'] != .9
            or receipt['status'] != 'api_returned_true' or not receipt['moving_handoff_passed']
            or not request['id'] == receipt['id'] == activation['request_id']
            or not request['student_owner_pid'] == receipt['student_owner_pid']
                == activation['student_owner_pid']
            or before['generation'] != after['generation'] or before['epoch'] != after['epoch']
            or any(s['car_id'] != 'bmw_z4_gt3' or s['track_id'] != 'monza'
                   or s['is_online'] for s in (before, after))
            or before['paused'] or not before['live'] or before['ai_controlled']
            or not 16 <= before['speed_kmh'] <= 18.5 or before['wheels_outside'] != 0
            or not after['paused'] or after['ai_controlled'] or after['speed_kmh'] >= .1
            or after['gas'] >= .02 or after['brake'] <= .98
            or receipt['maximum_speed_kmh'] > 20
            or receipt['maximum_position_delta_m'] > (180 if duration == 30 else 20)
            or [r['name'] for r in receipt['api_calls']] != expected_calls
            or not all(r['returned'] for r in receipt['api_calls'])):
        raise ValueError('Pinned fixed moving handoff, API order and native stop required')
    events = [json.loads(line) for line in
              (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    with load_array(root/'handoff-observations.npz') as packed:
        saved = {key: packed[key] for key in packed.files}
    if (saved['teacher_phase'].dtype != np.bool_
            or any(len(v) != len(saved['observation_time']) for v in saved.values())):
        raise ValueError('Exact observation lengths and boolean phase roles required')
    records, pairs = handoff_observation_records(events, saved['teacher_phase'])
    for event in events:
        if event['kind'] in ('raw_pair', 'expert_raw_native'):
            sources = [('native', 'prefix')]
            if event['kind'] == 'raw_pair':
                sources.append(('sdk', 'wire'))
            for source, name in sources:
                raw = base64.b64decode(event[source][name+'_base64'], validate=True)
                if hashlib.sha256(raw).hexdigest() != event[source][name+'_sha256']:
                    raise ValueError('Original raw SDK/native hash differs')
    independent = None
    if config['arguments'].get('handoff_native_receipts'):
        _, independent = verified_handoff_native_receipts(root, events)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862990,
        events_sha256=sha256(root/'events.jsonl'),
        observations_sha256=sha256(root/'handoff-observations.npz'),
        prefix_audit_manifest_sha256=sha256(prefix/'manifest.json'),
        expert_receipt_sha256=sha256(root/'expert-handoff/api-receipt.json'),
        independent_native_receipt_capture=independent))
    errors, effective, label_window_candidates = {}, 0, 0
    release = next(e for e in events if e['kind'] == 'handoff_vjoy_released')
    retina = CSPCameraRetina(config['pixel_configuration'])
    checkpoint = Path(config['arguments']['training_run'])/'checkpoint.pt'
    if sha256(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Original model checkpoint changed')
    torch.set_num_threads(1)
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in learned['graph'].items()}
    model = MultiViewCircuitController(graph, **learned['constructor']).eval().requires_grad_(False)
    model.load_state_dict(learned['model'])
    state, neural_error = model.initial_state(), 0.
    with CameraArchive(root/'vision') as camera:
        if camera.bundle_sha256 != result['camera_archive_receipt']['bundle_sha256']:
            raise ValueError('Camera bundle receipt changed')
        camera.verify_all()
        initial = camera.json('clock-initial.json')
        clock = CSPClockBounds(initial['generation'],
            maximum_width=initial['maximum_width_seconds'],
            host_quantization=initial.get('host_clock_quantization_seconds', 0.))
        for ping in initial['exchanges']:
            clock.observe(ping['sent'], ping['received'], ping['producer_clock'],
                          initial['generation'])
        if clock.record() != initial or clock.generation != before['generation']:
            raise ValueError('Original same-generation clock bounds required')
        ai_low, ai_high = fixed_hint_window(receipt['telemetry'], 15, clock.low, clock.high)
        for row in records:
            native_record = pairs[row['native_packet_id']]
            native = decode_physics_prefix(base64.b64decode(native_record['prefix_base64']),
                                             native_record['received_monotonic_seconds'])
            item, samples = row['camera'], {}
            frame_bytes = camera.read(Path(item['file']).name)
            frame = json.loads(frame_bytes)
            verify_shared_frame(frame, item,
                camera.json(f'ack-{item["capture_id"]:04d}.json')
                if frame.get('transport') == 'shared_immutable_v1' else None)
            if (hashlib.sha256(frame_bytes).hexdigest() != item['file_sha256']
                    or frame['index'] != item['capture_id']
                    or native.packet_id != row['native_packet_id']
                    or frame['generation'] != clock.generation):
                raise ValueError('Camera generation changed')
            sources = {view['view_id']: view for view in frame['views']}
            for view in item['views']:
                value = camera.read(Path(view['file']).name)
                pixels = np.array(Image.open(io.BytesIO(value)).convert('RGB'))
                lower = clock.render_start_lower(sources[view['view_id']]['render_started_clock'],
                                                  clock.generation)
                if (hashlib.sha256(value).hexdigest() != view['file_sha256']
                        or hashlib.sha256(pixels.tobytes()).hexdigest() != view['rgb_sha256']
                        or lower != view['render_started_lower']):
                    raise ValueError('Raw view bytes/causal source time changed')
                samples[view['view_id']] = dict(rgb=pixels, capture_id=item['capture_id'],
                    generation=clock.generation, received=item['received'],
                    render_started_lower=lower)
            packed = retina.pack(samples, decision_time=row['decision_monotonic_seconds'],
                native_scalars={k: native.fields[k] for k in retina.required_native_fields},
                native_receipt=native.received_monotonic_seconds)
            index = row['index']
            if saved['observation_time'][index] != row['decision_monotonic_seconds']:
                raise ValueError('Observation timestamp changed')
            for key, value in packed.items():
                original = saved['obs_'+key][index]
                errors[key] = max(errors.get(key, 0), float(np.max(np.abs(value-original))))
                if not np.array_equal(value, original):
                    raise ValueError('Full prefix/suffix sensory replay differs: '+key)
            if not row['expert']:
                drive = model.prepare_drive({k: torch.from_numpy(v)[None]
                                             for k, v in packed.items()})
                for _ in range(4):
                    prediction, state = model.advance(drive, state)
                error = float(np.max(np.abs(prediction[0].numpy()-row['prediction'])))
                neural_error = max(neural_error, error)
                if error > 1e-7:
                    raise ValueError('Prefix/waiting neural state did not replay without reset')
            if row['expert']:
                effective += int(effective_motion_after_release(
                    native, release.get('released_at', release['host_monotonic_seconds'])))
                label_window_candidates += int(
                    ai_low <= row['decision_monotonic_seconds'] <= ai_high)
    if effective < 2 or effective != result['effective_ai_motion_samples_after_release']:
        raise ValueError('Independent post-release effective AI motion count differs')
    stopped = next(e for e in reversed(events) if e['kind'] == 'post_stop_feedback')['diagnostic']
    raw_stop = pairs[stopped['native_packet_id']]
    stop = decode_physics_prefix(base64.b64decode(raw_stop['prefix_base64']),
                                  raw_stop['received_monotonic_seconds'])
    if (abs(stop.fields['speedKmh']) >= .1 or stop.fields['gas'] >= .02
            or stop.fields['brake'] <= .98):
        raise ValueError('Original native stop is not independently verified')
    intervals = np.diff(saved['observation_time'])
    write_json(run/'result.json', dict(status='completed', observations=len(records),
        student_prefix_observations=sum(not r['expert'] and not r['bridge'] for r in records),
        waiting_observations=sum(r['bridge'] for r in records),
        prefix_and_waiting_neural_replay_max_error=neural_error,
        waiting_predictions_applied=False,
        expert_suffix_observations=int(saved['teacher_phase'].sum()),
        actor_replay_max_errors=errors, same_retina_history_across_handoff=True,
        student_model_prefix_exact_replay=True, effective_ai_motion_samples_after_release=effective,
        expert_observations_inside_conservative_ai_window=label_window_candidates,
        teacher_phase_is_intent_not_label_mask=True, labels_created=False,
        maximum_observation_interval_seconds=float(intervals.max()),
        intervals_above80ms=int((intervals > .08).sum()), native_stop_verified=True,
        independent_native_receipt_capture=independent,
        autonomous_qualification=False, full_lap_verified=False,
        continuous40ms_training_admissible=False, ui_actions_sent=0,
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
