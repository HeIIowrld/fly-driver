"""Reproduce native fly requests from recorded pose-free pixels/body; no live interfaces."""

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
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_pixel_retina import NativePixelRetina
from flybrain_drive.ac_route_progress import RouteProgress
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes
from flybrain_drive.multiview_sensory import MultiViewPacker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seal-observations', action='store_true',
                        help='Export replayed actor observations for offline expert queries')
    args = parser.parse_args()
    root = Path(args.trial)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    recorded = json.loads((root / 'result.json').read_text(encoding='utf8'))
    config = manifest['config']
    checkpoint = Path(config['arguments']['training_run']) / 'checkpoint.pt'
    if sha256(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Trial model checkpoint changed')
    events_file = root / 'events.jsonl'
    events = [json.loads(line) for line in events_file.read_text(encoding='utf8').splitlines()]
    requests = [r for r in events if r['kind'] == 'student_request']
    feedback = {r['index']: r for r in events if r['kind'] == 'feedback'}
    if (not requests or len(requests) != recorded['student_requests']
            or [r['index'] for r in requests] != list(range(len(requests)))
            or set(feedback) != set(range(len(requests)))):
        raise ValueError('Exactly one recorded feedback per contiguous student request required')
    pairs, sdk_sequences = {}, {}
    for event in events:
        if event['kind'] != 'raw_pair':
            continue
        for name, tag in [('sdk', 'wire'), ('native', 'prefix')]:
            raw = base64.b64decode(event[name][tag + '_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != event[name][tag + '_sha256']:
                raise ValueError('Recorded raw sensor hash differs')
        key = event['native']['packet_id']
        if key in pairs:
            raise ValueError('Duplicate native sample identity')
        pairs[key] = event['native']
        sdk_sequences[key] = event['sdk']['sequence']
    calibration_path = Path(config['arguments']['calibration_result']).parent / 'samples.jsonl'
    calibration_rows = [json.loads(line) for line in
                        calibration_path.read_text(encoding='utf8').splitlines()]
    stages = [[] for _ in range(4)]
    for row in calibration_rows:
        if 'native' not in row:
            continue
        raw = base64.b64decode(row['native']['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != row['native']['prefix_sha256']:
            raise ValueError('Independent stationary calibration prefix hash differs')
        sample = decode_physics_prefix(raw, row['native']['received_monotonic_seconds'])
        stages[row['stage']].append(sample.fields['steerAngle'])
    calibration = json.loads((root / 'calibration-input.json').read_text(encoding='utf8'))
    if any(len(stage) != 8 for stage in stages):
        raise ValueError('Four independent eight-sample calibration stages required')
    medians = np.array([np.median(stage) for stage in stages])
    native_steering_gain = float((medians[1]-medians[2]) / (2*calibration['request_amplitude']))
    if (not native_steering_gain > 0 or max(abs(medians[[0, 3]])) > .001):
        raise ValueError('Independent native steering calibration differs')
    torch.set_num_threads(1)
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    csp = config['arguments'].get('camera_backend') == 'csp'
    retina = CSPCameraRetina(config['pixel_configuration']) if csp else \
        NativePixelRetina(config['pixel_configuration'])
    camera_clock = None
    archive = None
    if csp:
        archive = CameraArchive(root/'vision')
        expected_bundle = recorded.get('camera_archive_receipt', {}).get('bundle_sha256')
        if expected_bundle is not None and archive.bundle_sha256 != expected_bundle:
            raise ValueError('Camera bundle differs from trial receipt')
        archive.verify_all()
        recorded_clock = archive.json('clock-initial.json')
        camera_clock = CSPClockBounds(recorded_clock['generation'],
            maximum_width=recorded_clock['maximum_width_seconds'],
            host_quantization=recorded_clock.get('host_clock_quantization_seconds', 0.))
        for ping in recorded_clock['exchanges']:
            camera_clock.observe(ping['sent'], ping['received'], ping['producer_clock'],
                                 recorded_clock['generation'])
        if camera_clock.record() != recorded_clock:
            raise ValueError('Recorded CSP clock bounds do not reproduce')
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in learned['graph'].items()}
    model = MultiViewCircuitController(graph, **learned['constructor'])
    model.load_state_dict(learned['model'])
    model.eval().requires_grad_(False)
    state = model.initial_state()
    replay_errors, feedback_errors, native_errors, detailed = [], [], [], []
    packed_rows, quality, decision_times = [], [], []
    with torch.no_grad():
        for request in requests:
            meta = request['actor_input']
            native = pairs[meta['native_packet_id']]
            if meta['native_prefix_sha256'] != native['prefix_sha256']:
                raise ValueError('Actor native input does not match raw sensor recording')
            sample = decode_physics_prefix(base64.b64decode(native['prefix_base64']),
                                            native['received_monotonic_seconds'])
            image = meta['image']
            if csp:
                file = Path(image['file'])
                if file.parts[:-1] != ('vision',):
                    raise ValueError('Actor image path differs')
                metadata_bytes = archive.read(file.name)
                if hashlib.sha256(metadata_bytes).hexdigest() != image['file_sha256']:
                    raise ValueError('Actor image hash differs')
                frame = json.loads(metadata_bytes)
                verify_shared_frame(frame, image,
                    archive.json(f'ack-{image["capture_id"]:04d}.json')
                    if frame.get('transport') == 'shared_immutable_v1' else None)
                if frame['generation'] != camera_clock.generation:
                    raise ValueError('CSP camera producer generation changed')
                sources = {v['view_id']: v for v in image['views']}
                camera_samples = {}
                for view in frame['views']:
                    name, source = view['view_id'], sources[view['view_id']]
                    pixels_path = Path(source['file'])
                    if pixels_path.parts[:-1] != ('vision',):
                        raise ValueError('CSP image path differs')
                    image_bytes = archive.read(pixels_path.name)
                    if hashlib.sha256(image_bytes).hexdigest() != source['file_sha256']:
                        raise ValueError('CSP source pixel file changed')
                    with Image.open(io.BytesIO(image_bytes)) as data:
                        pixels = np.array(data.convert('RGB'))
                    lower = camera_clock.render_start_lower(view['render_started_clock'],
                                                             frame['generation'])
                    if (hashlib.sha256(pixels.tobytes()).hexdigest() != source['rgb_sha256']
                            or lower != source['render_started_lower']):
                        raise ValueError('CSP pixel bytes or lower clock bound changed')
                    camera_samples[name] = dict(rgb=pixels, capture_id=image['capture_id'],
                        generation=frame['generation'], received=image['received'],
                        render_started_lower=lower)
            else:
                file = (root/image['file']).resolve()
                if not file.is_relative_to(root.resolve()) or sha256(file) != image['file_sha256']:
                    raise ValueError('Actor image path/hash differs')
                rgb = load_array(file, allow_pickle=False)
                if hashlib.sha256(rgb.tobytes()).hexdigest() != image['rgb_sha256']:
                    raise ValueError('Actor image pixels differ')
            fields = {k: sample.fields.get(k) for k in retina.required_native_fields}
            if (set(meta['body_fields']) != set(fields)
                    or request['teacher_action_used'] or request['diagnostic_pose_used_by_actor']):
                raise ValueError('Actor sensor/teacher separation differs')
            if not csp and (image['callback_finished_monotonic_seconds']
                            > sample.received_monotonic_seconds):
                raise ValueError('Noncausal image candidate')
            if csp:
                packed = retina.pack(camera_samples,
                    decision_time=meta['decision_monotonic_seconds'],
                    native_scalars=fields, native_receipt=sample.received_monotonic_seconds)
            else:
                packed = retina.pack(rgb, capture_id=image['capture_id'],
                    callback_time=image['callback_finished_monotonic_seconds'],
                    decision_time=meta['decision_monotonic_seconds'], native_scalars=fields,
                    native_receipt=sample.received_monotonic_seconds)
            if set(meta['actor_fields']) != set(packed):
                raise ValueError('Actor tensor fields differ')
            if args.seal_observations:
                packed_rows.append(packed)
                decision_times.append(meta['decision_monotonic_seconds'])
                quality.append(dict(index=request['index'],
                    source_native_packet_id=sample.packet_id,
                    source_sdk_sequence=sdk_sequences[sample.packet_id],
                    source_camera_capture_id=image['capture_id'],
                    camera_generation=image.get('generation'),
                    image_sha256=hashlib.sha256(json.dumps(
                        [v['rgb_sha256'] for v in image['views']] if csp
                        else [image['rgb_sha256']]).encode()).hexdigest(),
                    all_views_valid=bool(packed['view_valid'].all())))
            drive = model.prepare_drive({k: torch.from_numpy(v)[None] for k, v in packed.items()})
            for _ in range(4):
                prediction, state = model.advance(drive, state)
            value = prediction[0].numpy()
            replay_errors.append(np.abs(value - np.asarray(request['prediction'])))
            expected = value.copy()
            expected[0] *= request['steering_request_scale']
            actual_request = request['request']['requested_control']
            encoded = np.array([actual_request[k] for k in ('steering', 'throttle', 'brake')])
            if not np.allclose(expected, encoded, atol=1e-7, rtol=1e-6):
                raise ValueError('Actuator request differs from unblended student prediction')
            applied = feedback[request['index']]['diagnostic']
            # SDK steering scale is the same-session measured 450/device request, not roadwheel rad.
            applied_control = [applied['sdk_steer_raw'], applied['sdk_throttle'],
                               applied['sdk_brake']]
            expected_sdk = encoded * [450., 1., 1.]
            feedback_errors.append(np.abs(np.array(applied_control) - expected_sdk))
            next_native = pairs[applied['native_packet_id']]
            next_sample = decode_physics_prefix(base64.b64decode(next_native['prefix_base64']),
                                                next_native['received_monotonic_seconds'])
            native_control = np.array([next_sample.fields[k] for k in
                                       ('steerAngle', 'gas', 'brake')])
            native_errors.append(np.abs(native_control-encoded*[native_steering_gain, 1., 1.]))
            detailed.append(dict(index=request['index'], reproduced_prediction=value.tolist(),
                requested=encoded.tolist(), measured_sdk=applied_control,
                measured_native=native_control.tolist()))
    if archive is not None:
        archive.close()
    replay_max = np.max(replay_errors, axis=0)
    if np.any(replay_max > 1e-6):
        raise ValueError('Recorded student inference does not reproduce')
    diagnostics = [e['diagnostic'] for e in events if e['kind'] in
                   ('feedback', 'post_stop_feedback')]
    preflight = json.loads((root / 'preflight.json').read_text(encoding='utf8'))['diagnostic']
    origin = np.array([preflight['x'], preflight['y']])
    displacement = max(float(np.linalg.norm(np.array([d['x'], d['y']]) - origin))
                       for d in diagnostics)
    route_evidence = None
    if config.get('planned_travel_metric') == 'sdk_route':
        origins = [event for event in events if event['kind'] == 'route_origin']
        if len(origins) != 1:
            raise ValueError('Exactly one original route origin is required')
        original = origins[0]['diagnostic']
        key = original['native_packet_id']
        if key not in pairs or sdk_sequences[key] != original['sdk_sequence']:
            raise ValueError('Route origin lacks the original recorded sensor pair')
        progress = RouteProgress(original, maximum_speed_kmh=config['max_speed_kmh'])
        route_error = None
        try:
            for index in range(len(requests)):
                progress.observe(feedback[index]['diagnostic'])
        except ValueError as error:
            route_error = str(error)
        reached = route_error is None and progress.distance_m >= config['planned_travel_m']
        claimed = recorded.get('termination') == 'planned_travel_reached'
        if claimed and (not reached or not np.isclose(progress.distance_m,
                recorded['measured_travel_m'], rtol=0, atol=1e-5)):
            raise ValueError('Claimed route target does not reproduce from original feedback')
        route_evidence = dict(distance_m=progress.distance_m,
            target_m=config['planned_travel_m'], numeric_target_reached=reached,
            continuity_error=route_error, claimed_target_reproduced=claimed and reached,
            full_lap_verified=False, scope='SDK route counters plus bounded geometry/time; '
                'not a full-lap certificate or independent engine-session proof')
    maximum_errors = np.max(feedback_errors, axis=0)
    # Report one-row publication lag separately; do not silently realign zero-lag failures.
    delayed_sdk_errors = []
    for previous, next_row in zip(detailed[:-1], detailed[1:], strict=True):
        delayed_sdk_errors.append(np.abs(np.array(next_row['measured_sdk'])
                                         - np.array(previous['requested'])*[450., 1., 1.]))
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        trial_events_sha256=sha256(events_file), checkpoint_sha256=sha256(checkpoint),
        seed=learned['training_configuration']['seed'], actuator_messages_sent=0))
    write_json(run / 'reproduced-requests.json', detailed)
    if args.seal_observations:
        stops = [e for e in events if e['kind'] == 'post_stop_feedback']
        if (not stops or not recorded.get('post_stop_verified')
                or recorded['final_vjoy_status'] != 1):
            raise ValueError('Verified terminal stop and released driver required before sealing')
        stop = pairs[stops[-1]['diagnostic']['native_packet_id']]
        stopped = decode_physics_prefix(base64.b64decode(stop['prefix_base64']),
                                        stop['received_monotonic_seconds'])
        if (abs(stopped.fields['speedKmh']) >= .1 or stopped.fields['gas'] >= .02
                or stopped.fields['brake'] <= .98):
            raise ValueError('Raw terminal native prefix does not verify stop')
        dataset = run / 'dataset'
        dataset.mkdir()
        episode = dataset / 'student-observations.npz'
        np.savez_compressed(episode,
            **{'obs_'+k: np.stack([row[k] for row in packed_rows])
               for k in MultiViewPacker.fields},
            action=np.array([r['prediction'] for r in requests], np.float32),
            action_valid=np.ones((len(requests), 3), bool),
            observation_time=np.array(decision_times, np.float64))
        write_json(dataset / 'dataset.json', dict(schema='flybrain.multiview.teacher.v1',
            source='exactly replayed actual student observations; no expert labels',
            source_request_kind='student', action_profile='normalized_ac_absolute',
            decisions=len(requests),
            action_provider='student', expert_labels_available=False,
            labels_are_executed_requests=True,
            action_semantics='student canonical requests; device steering transform excluded',
            sensory_calibration=retina.calibration, actor_fields=sorted(MultiViewPacker.fields),
            raw_events_file='events.jsonl', raw_events_sha256=sha256(events_file),
            native_AC_training_data=True, physical_optics_calibrated=False,
            exposure_synchronization_verified=False, decision_dt_seconds=.04,
            brain_ticks_per_decision=4,
            episodes=[dict(episode_id='student-observations', file=episode.name,
                           sha256=sha256(episode), steps=len(requests))]))
        write_json(run / 'sample-quality.json', quality)
        load_episodes(dataset, retina.calibration)
    write_json(run / 'result.json', dict(status='completed', student_requests=len(requests),
        exact_model_pixel_body_replay_verified=True, replay_max_absolute_error=replay_max.tolist(),
        actor_input_pose_free=True,
        teacher_actions_used=bool(recorded.get('teacher_action_used', False)),
        prefix_actor_teacher_actions_used=False,
        audit_scope='student requests only; expert suffix requires separate audit',
        max_sdk_response_absolute_error=maximum_errors.tolist(),
        max_native_response_absolute_error=np.max(native_errors, axis=0).tolist(),
        native_steering_calibration_gain=native_steering_gain,
        native_steering_calibration_source_sha256=sha256(calibration_path),
        native_steering_units='measured game signal only; roadwheel radians unverified',
        one_row_delayed_sdk_max_absolute_error=np.max(delayed_sdk_errors, axis=0).tolist()
            if delayed_sdk_errors else None,
        sdk_lag_scope='consecutive-decision diagnostic; physical publication delay uncalibrated',
        sdk_response_within_declared_tolerance=bool(np.all(maximum_errors <= [.2, .005, .005])),
        sdk_response_tolerance=[.2, .005, .005],
        response_scope='next sampled SDK response, not exact actuator application timing',
        maximum_recorded_displacement_m=displacement,
        route_progress_evidence=route_evidence,
        maximum_recorded_speed_kmh=max(abs(d['speed_kmh']) for d in diagnostics),
        foreground_events=sum(e.get('was_foreground', False) for e in events),
        actual_post_stop_verified=recorded.get('post_stop_verified', False),
        final_vjoy_status=recorded['final_vjoy_status'],
        trial_termination=recorded['status'], termination_reason=recorded.get('reason'),
        trial_profile=config['arguments'].get('trial_profile', 'first'),
        planned_travel_target_m=config.get('planned_travel_m'),
        planned_travel_metric=config.get('planned_travel_metric', 'displacement'),
        planned_travel_reached=recorded.get('termination') == 'planned_travel_reached'
            if config.get('planned_travel_m') is not None else None,
        path_following_or_lap_success_verified=False, full_multiview_ready=False,
        calibrated_optics=False, exposure_synchronization_verified=False,
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
