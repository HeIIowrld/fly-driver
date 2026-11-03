"""Read-only preflight by default; --execute permits a finite background SAC teacher segment."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_geometric_teacher import GeometricRecoveryTeacher, run_geometric_recovery
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_static_identity_prefix
from flybrain_drive.ac_runtime_gc import defer_automatic_cyclic_gc
from flybrain_drive.ac_sac_teacher import PrivilegedSACTeacher, TeacherReferenceCurve
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.ac_teacher_observation import PrivilegedTeacherObservationBuilder
from flybrain_drive.ac_teacher_pilot import (
    check_pilot_state,
    run_teacher_segment,
    validate_calibration,
)
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_teacher_vision import TeacherVisionRecorder
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    parser.add_argument('--calibration-result',
                        help='Same-session steering calibration result.json')
    parser.add_argument('--decisions', type=int, default=50)
    parser.add_argument('--max-speed-kmh', type=float, default=60.,
                        help='Explicit virtual-track trial ceiling, at most 100 km/h')
    parser.add_argument('--collection-speed-kmh', type=float,
                        help='Teacher-only pedal governor; at least 10 below hard speed ceiling')
    parser.add_argument('--teacher-kind', choices=['sac', 'geometric_recovery'], default='sac')
    parser.add_argument('--geometric-target-speed-kmh', type=float, default=20.,
                        help='Geometric teacher task maximum, 5..30km/h; hard ceiling >= target+10')
    parser.add_argument('--kinematic-calibration',
                        help='Measured teacher-only steering/curvature fit')
    parser.add_argument('--steering-request-scale', type=float, default=1.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--camera-backend', choices=['window', 'csp'], default='window')
    parser.add_argument('--retina-config', default='configs/ac-csp-retina.json',
                        help='Explicit CSP sensory registration for the recorded actor arrays')
    parser.add_argument('--require-background', action='store_true',
                        help='Abort if engine is foreground; never changes window focus')
    parser.add_argument('--camera-archive-format', choices=['files', 'bundle'], default='files')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    host_clock = time.perf_counter if args.camera_backend == 'csp' else time.monotonic
    if not 1 <= args.decisions <= 500:
        raise ValueError('Pilot requires 1..500 decisions (up to 20 seconds nominal)')
    if not 0 < args.max_speed_kmh <= 100:
        raise ValueError('Finite pilot speed ceiling in (0, 100] required')
    if args.collection_speed_kmh is not None and not (
            10 <= args.collection_speed_kmh <= args.max_speed_kmh-10):
        raise ValueError('Collection target requires 10km/h headroom below the hard ceiling')
    config = json.loads(Path(args.axis_config).read_text(encoding='utf8'))
    sensory_config = json.loads(Path(args.retina_config).read_text(encoding='utf8')) \
        if args.camera_backend == 'csp' else None
    codec = json.loads(Path('configs/ac-sac-replay.json').read_text(encoding='utf8'))
    kinematic = None
    if args.teacher_kind == 'geometric_recovery':
        if not 5 <= args.geometric_target_speed_kmh <= 30 or (
                not args.geometric_target_speed_kmh+10 <= args.max_speed_kmh <= 60):
            raise ValueError('Geometric task needs 5..30km/h and explicit 10km/h headroom')
        if args.collection_speed_kmh is not None or not args.kinematic_calibration:
            raise ValueError('Recovery uses its own speed task and requires explicit measured fit')
        kinematic = json.loads(Path(args.kinematic_calibration).read_text(encoding='utf8'))
        if kinematic['status'] != 'completed' or not kinematic['teacher_only']:
            raise ValueError('Completed teacher-only kinematic fit required')
    geo = json.loads(Path('configs/ac-teacher-geometry-result.json').read_text())[
        'geometry_configuration']
    torch.set_num_threads(1)
    run = begin_run(args.output, dict(arguments=vars(args), axis_configuration=config,
        teacher_configuration=codec, geometry_configuration=geo, seed=codec['seed'],
        scope=__doc__, ui_actions_sent=0, fly_actor_driving=False,
        image_recording_on_execute=True, exposure_synchronization_verified=False,
        kinematic_calibration=kinematic, csp_sensory_configuration=sensory_config))
    reader, status_reader, session, vision = None, None, None, None
    receipt_bounds = {}
    native_samples, actor_observations, actor_times = {}, [], []
    gc_events = []

    def trace_gc(phase, info):
        if len(gc_events) < 4096:
            gc_events.append(dict(phase=phase, generation=info['generation'],
                                  perf_counter_seconds=time.perf_counter()))

    gc.callbacks.append(trace_gc)
    actor_retina = CSPCameraRetina(sensory_config) if sensory_config else None
    result = dict(status='not_ready', actuator_messages_sent=0, goal_achieved=False)
    event_spool = (Path(os.environ['LOCALAPPDATA']) / 'flybrain-review-20261027'
                   / 'teacher-event-spool' / uuid.uuid4().hex / 'observations-and-requests.jsonl')
    event_spool.parent.mkdir(parents=True, exist_ok=False)
    write_json(run / 'event-spool.json', dict(file=str(event_spool), source_retained=True,
        archive_destination=str(run / event_spool.name),
        reason='Keep synchronous NAS writes outside the 40ms control cycle'))
    try:
        window_api = WindowsACWindow()
        windows = window_api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine window required')
        window = windows[0]
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / 'identity.json', identity)
        if (identity['car_model'] != 'bmw_z4_gt3' or identity['track'] != 'monza'
                or config['car_model'] != identity['car_model']
                or config['track'] != identity['track']):
            raise ValueError('Reviewed teacher requires BMW Z4 GT3 / Monza')
        if sha256(config['profile_path']) != config['profile_sha256']:
            raise ValueError('Active controls profile differs from reviewed axis configuration')
        if sha256(codec['checkpoint']) != codec['checkpoint_sha256']:
            raise ValueError('Public teacher checkpoint differs from reviewed source')
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                      timeout=.15, clock=host_clock)
        reader = NativePhysicsReader(timeout=.15, clock=host_clock)
        status_reader = NativeTeacherStatusReader()
        geometry = PrivilegedTrackGeometry(geo['track_csv'], geo['reference_csv'],
            track_sha256=geo['track_sha256'], reference_sha256=geo['reference_sha256'])
        builder = PrivilegedTeacherObservationBuilder(geometry,
            TeacherReferenceCurve(geo['reference_csv']), profile=codec['profile'],
            minimum_interval=.02, maximum_interval=.08)
        with event_spool.open('x', encoding='utf8') as stream:
            def emit(event):
                event = dict(event, host_monotonic_seconds=host_clock(),
                    was_foreground_at_record=bool(
                        window_api.user.GetForegroundWindow() == window.hwnd))
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()

            def guard():
                if args.require_background and window_api.user.GetForegroundWindow() == window.hwnd:
                    raise RuntimeError('Background-only teacher; no window focus action is sent')
                if (run / 'stop').exists():
                    raise RuntimeError('Owned run stop file detected')
                if (sha256(config['profile_path']) != config['profile_sha256']
                        or window_api.probe(window.hwnd, require_foreground=False) != window):
                    raise RuntimeError('Original engine window/profile changed')
                status = status_reader.snapshot()
                if status['builtin_ai_controlled']:
                    raise RuntimeError('Builtin AI owns the car controls')

            def receive():
                packet, physics = client.receive(), reader.receive()
                # Keep exact raw evidence even if assembly rejects a missing/stale field.
                emit(dict(kind='raw_pair', sdk=record_packet(packet),
                          native=physics_record(physics)))
                receipt_bounds[(packet.sequence, physics.packet_id)] = min(
                    packet.received_monotonic_seconds, physics.received_monotonic_seconds)
                if actor_retina is not None:
                    native_samples[physics.packet_id] = physics
                return builder.build(packet, physics)

            def reset_teacher_history(reason):
                nonlocal builder
                builder = PrivilegedTeacherObservationBuilder(geometry,
                    TeacherReferenceCurve(geo['reference_csv']), profile=codec['profile'],
                    minimum_interval=.02, maximum_interval=.08)
                emit(dict(kind='teacher_history_reset', reason=reason,
                          fabricated_rows=0, required_real_preceding_rows=3))

            guard()
            for _ in range(4):
                started = time.monotonic()
                current = receive()
                time.sleep(max(0, .04 - (time.monotonic() - started)))
            reasons = []
            try:
                if args.teacher_kind == 'geometric_recovery':
                    if (kinematic['profile_sha256'] != config['profile_sha256']
                            or kinematic['producer_session_id'] != current.producer_session_id):
                        raise ValueError('Kinematic fit must match this profile and engine session')
                    if abs(current.state['gap']) > 6.5:
                        raise ValueError('Recovery start gap exceeds 6.5m')
                    check_pilot_state(current, require_start=True,
                                     max_speed_kmh=args.max_speed_kmh,
                                     max_gap_m=max(3., abs(current.state['gap'])+1.5))
                else:
                    check_pilot_state(current, require_start=True,
                                      max_speed_kmh=args.max_speed_kmh)
            except ValueError as error:
                reasons.append(str(error))
            if args.calibration_result:
                calibration = json.loads(Path(args.calibration_result).read_text(encoding='utf8'))
                write_json(run / 'calibration-input.json', calibration)
                try:
                    validate_calibration(calibration, config, current.producer_session_id,
                                         steering_request_scale=args.steering_request_scale)
                except ValueError as error:
                    reasons.append(str(error))
            else:
                reasons.append('Same-session SDK steering calibration result required')
            controllers = inspect_vjoy_controllers()
            devices = controllers['vjoy_devices']
            if (controllers['candidate_ac_joy_index'] != 0 or len(devices) != 1
                    or devices[0]['instance_guid'] != config['instance_guid']):
                reasons.append('DirectInput identity/index differs from reviewed profile')
            write_json(run / 'preflight.json', dict(ready=not reasons, reasons=reasons,
                producer_session_id=current.producer_session_id, controllers=controllers,
                actor_input=False, images_collected=False, actuator_messages_sent=0))
            if reasons or not args.execute:
                result.update(status='not_ready' if reasons else 'preflight_only', reasons=reasons)
            else:
                teacher = PrivilegedSACTeacher(codec['checkpoint'], codec['checkpoint_sha256']) \
                    if args.teacher_kind == 'sac' else GeometricRecoveryTeacher(
                        geometry.line, kinematic['curvature_per_canonical_steering'],
                        maximum_speed_kmh=args.geometric_target_speed_kmh)
                sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
                session = VJoySession(sdk, config['device_id'],
                                      [AxisBinding(**b) for b in config['bindings']])
                spool = (Path(os.environ['LOCALAPPDATA']) / 'flybrain-review-20261027'
                         / 'pilot-spool' / uuid.uuid4().hex / 'vision')
                write_json(run / 'vision-spool.json', dict(directory=str(spool),
                    archive_destination=str(run / 'vision'), source_retained=True))
                vision = CSPVisionRecorder(window.hwnd, sensory_config,
                    max_frames=int(np.ceil(args.decisions*.04/.03))+80,
                    period=.04, image_format='bmp', shared_transport=True,
                    batched_shared_copy=True, async_frame_persistence=True,
                    pixel_preprocessor=actor_retina.precompute_pixels,
                    expected_source_sha256='78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6') \
                    if actor_retina else TeacherVisionRecorder(
                        spool, window.hwnd, max_frames=args.decisions+30)
                vision.wait_ready()
                # Collect before establishing the cadence-checked preceding rows.
                # Collection after those rows would create an artificial source gap.
                if gc.isenabled():
                    gc.collect(2)
                # Explicit new history after WGC startup, which may take several seconds.
                reset_teacher_history('vision_startup_completed')
                for attempt in range(5):
                    try:
                        for _ in range(4):
                            started = time.monotonic()
                            guard()
                            receive()
                            time.sleep(max(0, .04 - (time.monotonic() - started)))
                        break
                    except ValueError as error:
                        if (str(error) != 'Teacher source interval differs from declared decision cadence'
                                or attempt == 4):
                            raise
                        reset_teacher_history('stationary_source_cadence_retry')

                def image_context(current):
                    image_started = time.perf_counter()
                    candidate = vision.candidate_before(receipt_bounds[
                        (current.sdk_sequence, current.native_packet_id)])
                    if actor_retina is None:
                        return candidate
                    sample = native_samples[current.native_packet_id]
                    decision = host_clock()
                    fields = {key: sample.fields.get(key)
                              for key in actor_retina.required_native_fields}
                    visual_samples = vision.samples_for(candidate)
                    pack_started = time.perf_counter()
                    packed = actor_retina.pack(visual_samples,
                        decision_time=decision, native_scalars=fields,
                        native_receipt=sample.received_monotonic_seconds)
                    if not packed['view_valid'].all():
                        raise ValueError('CSP source became stale before teacher action')
                    packed_index = len(actor_observations)
                    actor_observations.append(packed)
                    actor_times.append(decision)
                    return dict(candidate, packed_observation_index=packed_index,
                        actor_decision_monotonic_seconds=decision,
                        actor_native_packet_id=sample.packet_id,
                        native_prefix_sha256=hashlib.sha256(sample.prefix_bytes).hexdigest(),
                        actor_body_field_names=sorted(fields),
                        image_pipeline_seconds=dict(cache_and_body=pack_started-image_started,
                            retina_pack=time.perf_counter()-pack_started),
                        diagnostic_pose_used_by_actor=False)

                # Both bounded runtimes release the device in their own finally blocks.
                # Move automatic cyclic collection outside that finite control scope.
                with defer_automatic_cyclic_gc() as gc_scope:
                    result['cyclic_gc_scope'] = gc_scope
                    if args.teacher_kind == 'geometric_recovery':
                        result.update(run_geometric_recovery(session, receive, teacher, guard, emit,
                            decisions=args.decisions, image_context=image_context,
                            steering_request_scale=args.steering_request_scale,
                            clock=time.perf_counter,
                            hard_speed_kmh=args.max_speed_kmh,
                            reset_history=reset_teacher_history))
                    else:
                        result.update(run_teacher_segment(
                            session, receive, teacher, codec, guard, emit,
                            decisions=args.decisions, period=.035, image_context=image_context,
                            max_speed_kmh=args.max_speed_kmh,
                            collection_speed_kmh=args.collection_speed_kmh,
                            steering_request_scale=args.steering_request_scale,
                            clock=time.perf_counter,
                            reset_history=reset_teacher_history))
    except Exception as error:
        result.update(status='aborted' if session and session.submit_attempts else 'not_ready',
                      error=type(error).__name__, reason=str(error))
    except BaseException as error:
        result.update(status='interrupted', error=type(error).__name__, reason=str(error))
        raise
    finally:
        gc.callbacks.remove(trace_gc)
        if session is not None:
            try:
                if session.owned:
                    write_json(run / 'last-neutral-request.json', session.close())
            except Exception as error:
                result['neutral_request_error'] = str(error)
            result['actuator_messages_sent'] = session.submit_attempts
        write_json(run / 'gc-events.json', gc_events)
        if reader is not None:
            reader.close()
        if status_reader is not None:
            status_reader.close()
        if vision is not None:
            try:
                result['vision'] = vision.close()
                receipt = (vision.archive_to(run / 'vision', bundle=True)
                           if actor_retina and args.camera_archive_format == 'bundle'
                           else vision.archive_to(run / 'vision'))
                result['vision']['archive_format'] = receipt.get('format', 'files')
                result['vision']['archive_receipt'] = receipt
                result['vision']['archive_hashes_verified'] = receipt['all_hashes_verified']
            except Exception as error:
                result['vision_cleanup_error'] = str(error)
        if actor_observations:
            np.savez_compressed(run / 'actor-observations.npz',
                **{'obs_'+key: np.stack([o[key] for o in actor_observations])
                   for key in actor_observations[0]}, observation_time=np.array(actor_times))
            result.update(actor_observations=len(actor_observations),
                actor_observations_sha256=sha256(run / 'actor-observations.npz'),
                actor_input_pose_free=True, camera_backend='csp')
            result.update(pixel_cache_hits=actor_retina.pixel_cache_hits,
                          pixel_cache_misses=actor_retina.pixel_cache_misses)
        if event_spool.exists():
            try:
                target = run / event_spool.name
                shutil.copy2(event_spool, target)
                if sha256(target) != sha256(event_spool):
                    raise ValueError('Local event spool/NAS archive hashes differ')
                result['events_archive_sha256'] = sha256(target)
                result['events_archive_verified'] = True
            except Exception as error:
                result['events_archive_error'] = str(error)
        result.update(ui_actions_sent=0, fly_actor_driving_verified=False,
            full_lap_verified=False, images_collected=bool(vision and vision.count),
            training_admissible=False, goal_achieved=False)
        write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
