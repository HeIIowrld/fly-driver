"""Default read-only preflight; --execute permits a finite low-speed fly student trial."""

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
from flybrain_drive.ac_expert_handoff import (
    OwnedExpertHandoff,
    effective_motion_after_release,
    validate_handoff_preset,
)
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_latest_tcp import LatestTcpTelemetryClient
from flybrain_drive.ac_native_clock import NativeGraphicsTimingReader
from flybrain_drive.ac_native_physics import (
    PHYSICS_LAYOUT,
    PHYSICS_PREFIX,
    NativePhysicsReader,
    decode_static_identity_prefix,
)
from flybrain_drive.ac_native_receipts import NativeReceiptCollector
from flybrain_drive.ac_pixel_retina import NativePixelRetina
from flybrain_drive.ac_student_pilot import (
    STUDENT_TRIAL_PROFILES,
    StudentTrialReading,
    check_trial_reading,
    has_applied_receipt_task,
    has_fixed_speed_task,
    run_student_trial,
)
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.ac_teacher_pilot import validate_calibration
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_teacher_vision import TeacherVisionRecorder
from flybrain_drive.ac_view_contract import validate_declared_live_view
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--retina-config')
    parser.add_argument('--camera-backend', choices=['window', 'csp'], default='window')
    parser.add_argument('--camera-transport', choices=['producer_file', 'shared'],
                        default='producer_file')
    parser.add_argument('--camera-archive-format', choices=['files', 'bundle'], default='files')
    parser.add_argument('--camera-wait-seconds', type=float, choices=[.02, .04], default=.02,
                        help='Bounded CSP wait; source age and total inference limits stay fixed')
    parser.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    parser.add_argument('--calibration-result', required=True)
    parser.add_argument('--steering-request-scale', type=float, default=.6)
    parser.add_argument('--decisions', type=int, default=20)
    parser.add_argument('--trial-profile',
                        choices=list(STUDENT_TRIAL_PROFILES),
                        default='first')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--expert-handoff', action='store_true',
                        help='Separate data probe: moving student to owned finite engine AI')
    parser.add_argument('--handoff-seconds', type=int, choices=[3, 30], default=3)
    parser.add_argument('--handoff-native-receipts', action='store_true',
                        help='Separate read-only100Hz native receipts during owned AI phase')
    parser.add_argument('--handoff-context', choices=['launch', 'curve200', 'curve320'],
                        default='launch')
    parser.add_argument('--diagnostic-snapshot', action='store_true',
                        help='Reuse unchanged SDK diagnostics only within original50ms receipt age')
    parser.add_argument('--local-event-spool', action='store_true',
                        help='Keep original event bytes locally; archive after control closes')
    parser.add_argument('--require-background', action='store_true',
                        help='Stop if AC is foreground; never change window focus')
    parser.add_argument('--record-lap-evidence', action='store_true',
                        help='Archive raw graphics timing beside SDK/native diagnostics')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    curve = args.handoff_context in ('curve200', 'curve320')
    handoff_minimum_route_m = {'launch': 0, 'curve200': 200, 'curve320': 320}[
        args.handoff_context]
    shared_transport = args.camera_transport == 'shared'
    if shared_transport and args.camera_backend != 'csp':
        raise ValueError('Shared texture transport requires the reviewed CSP renderer')
    if args.expert_handoff:
        validate_handoff_preset(args.handoff_context, args.decisions, args.handoff_seconds,
            args.handoff_native_receipts, args.trial_profile, args.camera_backend,
            args.require_background)
    if curve and not args.expert_handoff:
        raise ValueError('Curve context is only for the explicit owned expert data probe')
    if not args.expert_handoff and args.handoff_seconds != 3:
        raise ValueError('Extended handoff duration is only for explicit expert data probes')
    if args.handoff_native_receipts and not args.expert_handoff:
        raise ValueError('Independent receipt capture requires explicit expert handoff')
    if args.retina_config is None:
        args.retina_config = 'configs/ac-csp-retina.json' if args.camera_backend == 'csp' \
            else 'configs/ac-native-pixel-retina.json'
    tracking = args.trial_profile != 'first'
    limits = STUDENT_TRIAL_PROFILES[args.trial_profile]
    faster = args.trial_profile in ('tracking_30kmh', 'tracking_60m',
                                    'tracking_120m', 'tracking_240m', 'course_240m', 'course_480m',
                                    'game_valid_480m', 'game_valid_route_480m')
    host_clock = time.perf_counter if args.camera_backend == 'csp' else time.monotonic
    if not 1 <= args.decisions <= limits['decisions']:
        raise ValueError('Decision count exceeds declared student trial profile')
    speed_limit = limits['speed'] if tracking else 5.
    distance_limit = limits['displacement'] if tracking else .5
    planned_travel = limits['travel']
    camera_max_bytes = 1024**3 if args.trial_profile in (
        'tracking_240m', 'course_240m') else 512*1024**2
    if args.trial_profile in ('course_480m', 'game_valid_480m', 'game_valid_route_480m'):
        camera_max_bytes = 2*1024**3
    camera_max_bytes = limits.get('camera_bytes', camera_max_bytes)
    axis = json.loads(Path(args.axis_config).read_text(encoding='utf8'))
    training = Path(args.training_run)
    report = json.loads((training / 'result.json').read_text(encoding='utf8'))
    checkpoint = training / 'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified native training checkpoint required')
    torch.set_num_threads(1)
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = learned['training_configuration']
    receipt_task = limits.get('task') == 'applied_receipt'
    if receipt_task:
        if (args.camera_backend != 'csp' or not has_applied_receipt_task(
                learned.get('teacher_label_contract'), limits['task_speed'],
                args.steering_request_scale)):
            raise ValueError('Separate receipt trial requires its exact registered task and codec')
    elif tracking and not has_fixed_speed_task(learned.get('teacher_label_contract'),
                                               30. if faster else 20.):
        raise ValueError('Tracking validation requires a matching explicit fixed-speed checkpoint')
    if (config['action_profile'] != 'normalized_ac_absolute'
            or config['brain_ticks_per_decision'] != 4 or config['control_dt_seconds'] != .04):
        raise ValueError('Reviewed native codec and neural timing required')
    retina_type = CSPCameraRetina if args.camera_backend == 'csp' else NativePixelRetina
    retina = retina_type(json.loads(Path(args.retina_config).read_text(encoding='utf8')))
    if args.camera_backend == 'csp' and not learned.get('csp_input_registration'):
        raise ValueError('CSP camera trial requires explicitly registered model')
    live_view_contract = validate_declared_live_view(retina.config)
    profiles = live_view_contract.get('allowed_trial_profiles')
    if profiles is not None and args.trial_profile not in profiles:
        raise ValueError('Trial profile is not declared for the active camera transport')
    if retina.calibration != learned['constructor']['sensory_calibration']:
        raise ValueError('Live retina/body configuration differs from checkpoint')
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in learned['graph'].items()}
    model = MultiViewCircuitController(graph, **learned['constructor'])
    model.load_state_dict(learned['model'])
    model.eval().requires_grad_(False)
    # Initialize sparse-kernel caches without sensing or actions. Runtime state starts at zero.
    with torch.no_grad():
        warm_state = model.initial_state()
        for _ in range(4):
            _, warm_state = model.advance(torch.zeros_like(warm_state), warm_state)
    state = model.initial_state()
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=config['seed'],
        checkpoint_sha256=sha256(checkpoint), axis_configuration=axis,
        student_teacher_task_contract=learned.get('teacher_label_contract'),
        pixel_configuration=retina.config, max_speed_kmh=speed_limit,
        host_clock_implementation=time.get_clock_info(
            'perf_counter' if args.camera_backend == 'csp' else 'monotonic').implementation,
        declared_live_view=live_view_contract,
        csp_camera_wait_budget_seconds=args.camera_wait_seconds
            if args.camera_backend == 'csp' else 0.,
        csp_candidate_age_budget_seconds=min(retina.max_age, .11)
            if args.camera_backend == 'csp' else None,
        max_displacement_m=distance_limit, planned_travel_m=planned_travel,
        planned_travel_metric=limits.get('metric', 'displacement'),
        camera_max_bytes=camera_max_bytes,
        maximum_reference_gap_m=limits['gap'] if tracking else None,
        minimum_lateral_clearance_m=limits.get('clearance',
            1. if limits['gap'] is not None else None),
        maximum_tyres_out=limits.get('tyres', 2),
        require_valid_game_lap=limits.get('valid_lap', False),
        teacher_action_used=False, ui_actions_sent=0,
        kernel_warmup='four zero-drive ticks discarded before runtime initial state',
        recurrent_state='runtime initialized once; preserved between decisions'))
    events_file = run/'events.jsonl'
    event_source = events_file
    if args.local_event_spool:
        event_source = (Path(os.environ['LOCALAPPDATA'])/'flybrain-review-20261027'
                        / 'event-spool'/uuid.uuid4().hex/'events.jsonl')
        event_source.parent.mkdir(parents=True, exist_ok=False)
        write_json(run/'event-spool.json', dict(source=str(event_source),
            source_retained=True, original_event_clocks_preserved=True))
    result = dict(status='not_ready', actuator_messages_sent=0, ui_actions_sent=0,
                  student_requests=0, full_lap_verified=False, goal_achieved=False)
    gc_events = []

    def trace_gc(phase, info):
        if len(gc_events) < 4096:
            gc_events.append(dict(phase=phase, generation=info['generation'],
                                  perf_counter_seconds=time.perf_counter()))

    gc.callbacks.append(trace_gc)
    reader, status_reader, session, vision = None, None, None, None
    sdk, current_session = None, None
    graphics_reader = None
    lap_sdk_source = None
    sdk_source_proof = None
    handoff = None
    receipt_collector = None
    handoff_rows, handoff_times, handoff_roles = [], [], []
    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine window required')
        window = windows[0]
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / 'identity.json', identity)
        if (identity['car_model'] != 'bmw_z4_gt3' or identity['track'] != 'monza'
                or axis['car_model'] != identity['car_model']
                or axis['track'] != identity['track']):
            raise ValueError('Reviewed BMW Z4 GT3 / Monza required')
        controllers = inspect_vjoy_controllers()
        devices = controllers['vjoy_devices']
        if (controllers['candidate_ac_joy_index'] != 0 or len(devices) != 1
                or devices[0]['instance_guid'] != axis['instance_guid']):
            raise ValueError('Reviewed DirectInput identity/index required')
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                      timeout=.15, clock=host_clock)
        current_session = client.session_id
        reader = NativePhysicsReader(timeout=.15, clock=host_clock)
        status_reader = NativeTeacherStatusReader()
        if args.record_lap_evidence:
            graphics_reader = NativeGraphicsTimingReader()
            lap_sdk_source = LatestTcpTelemetryClient(clock=host_clock)
            if lap_sdk_source.session_id != current_session:
                raise ValueError('Lap evidence TCP producer differs from student telemetry')
        geometry = None
        if tracking:
            g = json.loads(Path('configs/ac-teacher-geometry-result.json').read_text(
                encoding='utf8'))['geometry_configuration']
            geometry = PrivilegedTrackGeometry(g['track_csv'], g['reference_csv'],
                track_sha256=g['track_sha256'], reference_sha256=g['reference_sha256'])
        with event_source.open('x', encoding='utf8') as stream, torch.no_grad():
            def emit(event):
                stream.write(json.dumps(dict(event, host_monotonic_seconds=host_clock(),
                    was_foreground=bool(native.user.GetForegroundWindow() == window.hwnd)),
                    allow_nan=False) + '\n')
                stream.flush()

            def guard():
                if args.require_background and native.user.GetForegroundWindow() == window.hwnd:
                    raise RuntimeError('Background-only trial: AC became foreground; '
                                       'no window focus action will be sent')
                if (run / 'stop').exists():
                    raise RuntimeError('Owned student trial stop file detected')
                if (native.probe(window.hwnd, require_foreground=False) != window
                        or sha256(axis['profile_path']) != axis['profile_sha256']):
                    raise ValueError('Original engine window or reviewed control profile changed')
                if status_reader.snapshot()['builtin_ai_controlled']:
                    raise ValueError('Builtin AI owns the vehicle')

            def receive():
                packet = client.snapshot() if args.diagnostic_snapshot else client.receive()
                sample = reader.receive()
                evidence = dict(kind='raw_pair', sdk=record_packet(packet),
                                native=physics_record(sample))
                if graphics_reader is not None:
                    try:
                        evidence['graphics'] = graphics_reader.snapshot().metadata()
                    except (OSError, RuntimeError, ValueError) as error:
                        evidence['graphics_error'] = str(error)
                emit(evidence)
                if (abs(packet.received_monotonic_seconds-sample.received_monotonic_seconds) > .05
                        or packet.raw['producer_session_id'] != current_session):
                    raise ValueError('Student diagnostic sensor identity/receipt skew changed')
                namespace = packet.raw['privileged_teacher']
                if (namespace['actor_input'] is not False
                        or namespace['protocol'] != 'flybrain.ac.privileged_teacher.v1'):
                    raise ValueError('Diagnostic namespace must remain explicitly teacher-only')
                truth = namespace['raw']
                values = PHYSICS_PREFIX.unpack(sample.prefix_bytes)
                raw, offset = {}, 0
                for name, _, width in PHYSICS_LAYOUT:
                    if width == 1:
                        raw[name] = values[offset]
                    offset += width
                diagnostic = dict(
                    speed_kmh=sample.fields['speedKmh'], gear=raw['gear'],
                    tyres_out=raw['numberOfTyresOut'], x=truth['world_position_x'],
                    lap_invalidated=truth.get('LapInvalidated'),
                    normalized_spline_position=truth.get('NormalizedSplinePosition'),
                    lap_count=packet.raw.get('LapCount'),
                    track_length_m=truth.get('track_length_m'),
                    y=truth['world_position_y'], sdk_steer_raw=packet.raw['steerAngle'],
                    sdk_throttle=packet.raw['accStatus'], sdk_brake=packet.raw['brakeStatus'],
                    native_throttle=sample.fields['gas'], native_brake=sample.fields['brake'],
                    native_steer_raw=sample.fields['steerAngle'],
                    native_received_monotonic_seconds=sample.received_monotonic_seconds,
                    sdk_received_monotonic_seconds=packet.received_monotonic_seconds,
                    control_feedback_contract='native game signals; SDK asynchronously published',
                    sdk_sequence=packet.sequence, native_packet_id=sample.packet_id)
                if geometry is not None:
                    features = geometry.features(diagnostic['x'], diagnostic['y'], -raw['heading'])
                    diagnostic.update(reference_gap_m=features['gap'],
                        minimum_lateral_clearance_m=float(min(features['sensors'][[0, -1]])))
                return StudentTrialReading(sample, diagnostic)

            guard()
            first = receive()
            check_trial_reading(first, require_start=True,
                                max_reference_gap_m=limits['gap'] if tracking else None,
                                min_lateral_clearance_m=limits.get('clearance'),
                                max_tyres_out=limits.get('tyres', 2),
                                require_valid_lap=limits.get('valid_lap', False))
            calibration = json.loads(Path(args.calibration_result).read_text(encoding='utf8'))
            validate_calibration(calibration, axis, current_session,
                                 steering_request_scale=args.steering_request_scale)
            write_json(run / 'calibration-input.json', calibration)
            sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
            if sdk.GetVJDStatus(axis['device_id']) != 1:
                raise ValueError('vJoy must be FREE; never take another owner')
            write_json(run / 'preflight.json', dict(ready=True,
                producer_session_id=current_session, diagnostic=first.diagnostic,
                native_gear_offset_verified=16, actuator_messages_sent=0))
            if not args.execute:
                result.update(status='preflight_only')
            else:
                spool = (Path(os.environ['LOCALAPPDATA']) / 'flybrain-review-20261027'
                         / 'student-pilot-spool' / uuid.uuid4().hex / 'vision')
                write_json(run / 'vision-spool.json',
                           dict(directory=str(spool), source_retained=True))
                camera_period = .04 if shared_transport else .03
                camera_frames = int(np.ceil(args.decisions*.04/camera_period))+50
                if args.expert_handoff and args.handoff_seconds == 30:
                    # Owned suffix plus teardown; existing3sec budget unchanged.
                    camera_frames += 825 if shared_transport else 1100
                vision = CSPVisionRecorder(window.hwnd, retina.config,
                    max_frames=camera_frames, max_bytes=camera_max_bytes,
                    pixel_preprocessor=retina.precompute_pixels, period=camera_period,
                    image_format='bmp' if shared_transport else 'png',
                    shared_transport=shared_transport) \
                    if args.camera_backend == 'csp' else \
                    TeacherVisionRecorder(spool, window.hwnd, max_frames=args.decisions+50)
                vision.wait_ready()
                if curve and vision.clock.high-vision.clock.low > .015:
                    raise ValueError('Curve handoff requires initial clock uncertainty<=15ms')
                if args.expert_handoff:
                    handoff = OwnedExpertHandoff(vision, run/'expert-handoff',
                                                 seconds=args.handoff_seconds)
                cached_key, rgb = None, None

                def infer(sample, *, teacher_phase=False, bridge_phase=False):
                    nonlocal cached_key, rgb, state
                    inference_started = time.perf_counter()
                    if args.camera_backend == 'csp':
                        candidate, selection_time, camera_wait = vision.wait_for_candidate(
                            max_age=min(retina.max_age, .11), timeout=args.camera_wait_seconds)
                    else:
                        selection_time, camera_wait = sample.received_monotonic_seconds, 0.
                        candidate = vision.candidate_before(selection_time, max_age=retina.max_age)
                    if args.camera_backend == 'window' and cached_key != candidate['file_sha256']:
                        file = spool / Path(candidate['file']).name
                        if sha256(file) != candidate['file_sha256']:
                            raise ValueError('Committed image file changed')
                        rgb = load_array(file, allow_pickle=False)
                        if hashlib.sha256(rgb.tobytes()).hexdigest() != candidate['rgb_sha256']:
                            raise ValueError('Committed image pixels changed')
                        cached_key = candidate['file_sha256']
                    fields = {k: sample.fields.get(k) for k in retina.required_native_fields}
                    stamp = host_clock()
                    pixels_ready = time.perf_counter()
                    if args.camera_backend == 'csp':
                        packed = retina.pack(vision.samples_for(candidate), decision_time=stamp,
                            native_scalars=fields, native_receipt=sample.received_monotonic_seconds)
                        image_oldest = min(v['render_started_lower'] for v in candidate['views'])
                        if not packed['view_valid'].all():
                            raise ValueError('CSP camera became stale before student inference')
                    else:
                        packed = retina.pack(rgb, capture_id=candidate['capture_id'],
                            callback_time=candidate['callback_finished_monotonic_seconds'],
                            decision_time=stamp, native_scalars=fields,
                            native_receipt=sample.received_monotonic_seconds)
                        image_oldest = candidate['callback_finished_monotonic_seconds']
                    if retina.wheel_spec is not None and not packed['organ_valid'][:18].reshape(
                            2, 9)[:, :7].all():
                        raise ValueError('Declared wheel proprioception unavailable before action')
                    if getattr(retina, 'yaw_spec', None) is not None and not packed[
                            'organ_valid'][126:128].all():
                        raise ValueError('Declared yaw body sensing unavailable before action')
                    if handoff is not None:
                        handoff_rows.append(packed)
                        handoff_times.append(stamp)
                        handoff_roles.append(teacher_phase)
                        if teacher_phase:
                            emit(dict(kind='expert_observation',
                                observation_index=len(handoff_rows)-1, camera=candidate,
                                native_packet_id=sample.packet_id,
                                decision_monotonic_seconds=stamp))
                            return
                    observation = {k: torch.from_numpy(v)[None] for k, v in packed.items()}
                    retina_done = time.perf_counter()
                    drive = model.prepare_drive(observation)
                    drive_done = time.perf_counter()
                    for _ in range(4):
                        prediction, state = model.advance(drive, state)
                    inferred = dict(prediction=prediction[0].numpy(),
                        oldest_input_monotonic_seconds=min(sample.received_monotonic_seconds,
                            image_oldest),
                        metadata=dict(image=candidate, body_fields=sorted(fields),
                            observation_index=len(handoff_rows)-1 if handoff is not None else None,
                            actor_fields=sorted(packed), native_packet_id=sample.packet_id,
                            native_prefix_sha256=hashlib.sha256(sample.prefix_bytes).hexdigest(),
                            visual_selection_monotonic_seconds=selection_time,
                            camera_wait_seconds=camera_wait,
                            decision_monotonic_seconds=stamp,
                            inference_pipeline_seconds=dict(camera=pixels_ready-inference_started,
                                retina=retina_done-pixels_ready,
                                sensory_drive=drive_done-retina_done,
                                brain_ticks=time.perf_counter()-drive_done)))
                    if bridge_phase:
                        emit(dict(kind='handoff_wait_observation',
                            prediction=inferred['prediction'].tolist(),
                            prediction_applied=False, actor_input=inferred['metadata']))
                    return inferred

                def observe_handoff_wait():
                    started = time.perf_counter()
                    waiting = receive()
                    infer(waiting.native, bridge_phase=True)
                    time.sleep(max(0., .04-(time.perf_counter()-started)))
                    return waiting

                def activate_handoff(owned, current):
                    emit(dict(kind='expert_handoff_requested'))
                    # Feedback is fresh and not yet packed; avoid waiting another SDK tick.
                    infer(current.native, bridge_phase=True)
                    guard()
                    check_trial_reading(current, max_speed_kmh=speed_limit,
                        max_tyres_out=limits.get('tyres', 2),
                        require_valid_lap=limits.get('valid_lap', False),
                        min_lateral_clearance_m=limits.get('clearance'))
                    return handoff.activate(owned, current, observe_wait=observe_handoff_wait)

                session = VJoySession(sdk, axis['device_id'],
                                      [AxisBinding(**b) for b in axis['bindings']])
                try:
                    result.update(run_student_trial(session, receive, infer, guard, emit,
                        decisions=args.decisions,
                        steering_request_scale=args.steering_request_scale,
                        trial_profile=args.trial_profile, max_speed_kmh=speed_limit,
                        max_displacement_m=distance_limit, schedule_clock=time.perf_counter,
                        clock=host_clock, expert_handoff=activate_handoff if handoff else None,
                        expert_handoff_minimum_route_m=handoff_minimum_route_m))
                    if handoff is not None:
                        if result['status'] != 'expert_handoff':
                            raise ValueError('Student ended before the separate moving handoff')
                        if session.owned or sdk.GetVJDStatus(axis['device_id']) != 1:
                            raise ValueError('vJoy not released after observed AI ownership')
                        released_at, effective_samples = host_clock(), 0
                        emit(dict(kind='handoff_vjoy_released', vjoy_status=1,
                                  released_at=released_at))
                        if args.handoff_native_receipts:
                            receipt_collector = NativeReceiptCollector()
                            receipt_collector.start()
                            emit(dict(kind='handoff_native_receipts_started',
                                      started_at=receipt_collector.started_at))
                        while handoff.poll() is None:
                            if receipt_collector is not None:
                                receipt_collector.check()
                            if time.monotonic() >= handoff.deadline:
                                raise TimeoutError('Same owned moving handoff deadline exceeded')
                            active_state = handoff.status()
                            if active_state.get('ai_controlled') and not active_state.get('paused'):
                                started = time.perf_counter()
                                try:
                                    # Teacher actor inputs do not consume SDK diagnostic truth.
                                    expert_sample = reader.receive()
                                    emit(dict(kind='expert_raw_native',
                                              native=physics_record(expert_sample)))
                                except TimeoutError:
                                    emit(dict(kind='expert_native_gap'))
                                    continue
                                infer(expert_sample, teacher_phase=True)
                                effective_samples += int(effective_motion_after_release(
                                    expert_sample, released_at))
                                time.sleep(max(0., .04-(time.perf_counter()-started)))
                            else:
                                time.sleep(.01)
                        handoff.finish()
                        result['effective_ai_motion_samples_after_release'] = effective_samples
                        if effective_samples < 2:
                            raise ValueError('No independent AI motion after vJoy release')
                        result.update(status='expert_handoff_completed',
                            data_collection_with_expert=True, student_closed_loop_trial=False,
                            expert_labels_available=False, training_admissible=False,
                            expert_observations=sum(handoff_roles))
                finally:
                    if handoff is not None and handoff.request is not None \
                            and handoff.terminal is None:
                        try:
                            handoff.finish(cancel=True)
                        except Exception as error:
                            result['handoff_cleanup_error'] = str(error)
                    if receipt_collector is not None:
                        capture = receipt_collector.close()
                        write_json(run/'expert-handoff/native-receipts.json',
                            [physics_record(sample) for sample in receipt_collector.samples])
                        capture['file_sha256'] = sha256(
                            run/'expert-handoff/native-receipts.json')
                        write_json(run/'expert-handoff/native-receipts-result.json', capture)
                        result['independent_native_receipts'] = capture
                        if capture['status'] != 'completed':
                            result.update(status='aborted',
                                          reason='Independent native receipt capture failed')
                    # Trial finally closes the device; measure the actual stopping response.
                    if not session.owned:
                        try:
                            deadline = time.monotonic() + 3.
                            while time.monotonic() < deadline:
                                stopped = receive()
                                emit(dict(kind='post_stop_feedback', diagnostic=stopped.diagnostic))
                                if (abs(stopped.diagnostic['speed_kmh']) < .1
                                        and stopped.native.fields['gas'] < .02
                                        and stopped.native.fields['brake'] > .98):
                                    result['post_stop_verified'] = True
                                    break
                                time.sleep(.05)
                        except Exception as error:
                            result['post_stop_error'] = str(error)
    except Exception as error:
        result.update(status='aborted', error=type(error).__name__, reason=str(error))
    except BaseException as error:
        result.update(status='interrupted', error=type(error).__name__, reason=str(error))
        raise
    finally:
        gc.callbacks.remove(trace_gc)
        if session is not None:
            if session.owned:
                try:
                    result['fallback_close'] = session.close()
                except Exception as error:
                    result['fallback_close_error'] = str(error)
            result['actuator_messages_sent'] = session.submit_attempts
        if sdk is not None:
            result['final_vjoy_status'] = sdk.GetVJDStatus(axis['device_id'])
        if reader is not None:
            reader.close()
        if status_reader is not None:
            status_reader.close()
        if graphics_reader is not None:
            graphics_reader.close()
        if lap_sdk_source is not None:
            try:
                lap_sdk_source.close()
                sdk_archive = run/'tcp-source-packets.jsonl'
                with sdk_archive.open('x', encoding='utf8') as stream:
                    for packet in lap_sdk_source.packets:
                        stream.write(json.dumps(record_packet(packet), allow_nan=False)+'\n')
                sdk_source_proof = dict(sha256=sha256(sdk_archive),
                    packets=len(lap_sdk_source.packets), receiver_terminated=True,
                    error=str(lap_sdk_source.error) if lap_sdk_source.error is not None else None)
            except (OSError, RuntimeError, ValueError) as error:
                sdk_source_proof = dict(receiver_terminated=False, error=str(error))
        if vision is not None:
            try:
                result['vision'] = vision.close()
                receipt = (vision.archive_to(run / 'vision', bundle=True)
                           if (args.camera_backend == 'csp'
                               and args.camera_archive_format == 'bundle')
                           else vision.archive_to(run / 'vision'))
                result['archive_hashes_verified'] = receipt['all_hashes_verified']
                result['camera_archive_receipt'] = receipt
            except Exception as error:
                result['vision_cleanup_error'] = str(error)
        if event_source != events_file and event_source.exists():
            try:
                with event_source.open('rb') as source, events_file.open('xb') as destination:
                    shutil.copyfileobj(source, destination)
                source_hash = sha256(event_source)
                if source_hash != sha256(events_file):
                    raise ValueError('Archived event bytes differ from local original')
                result['event_archive'] = dict(status='completed', source=str(event_source),
                    source_retained=True, sha256=source_hash,
                    bytes=event_source.stat().st_size, original_event_clocks_preserved=True)
            except Exception as error:
                result.update(status='aborted', event_archive_error=str(error),
                              reason='Original local events could not be exactly archived')
        if events_file.exists():
            events = [json.loads(line) for line in
                      events_file.read_text(encoding='utf8').splitlines()]
            result['student_requests'] = sum(e['kind'] == 'student_request' for e in events)
            if args.record_lap_evidence:
                raw_pairs = [e for e in events if e['kind'] == 'raw_pair']
                recorded = sum('graphics' in e for e in raw_pairs)
                result['lap_evidence'] = dict(raw_pairs=len(raw_pairs), graphics_records=recorded,
                    missing_graphics=len(raw_pairs)-recorded, events_sha256=sha256(events_file),
                    actor_input=False, official_validity_verified=False,
                    sdk_source=sdk_source_proof)
        result.update(diagnostic_pose_used_by_actor=False, teacher_action_used=False,
            exposure_sync_verified=False, multiview_ready=False, goal_achieved=False)
        if args.expert_handoff:
            result.update(teacher_action_used=handoff is not None and handoff.request is not None,
                          teacher_used_in_student_requests=False, autonomous_qualification=False)
            if handoff_rows:
                np.savez_compressed(run/'handoff-observations.npz',
                    **{'obs_'+k: np.stack([o[k] for o in handoff_rows]) for k in handoff_rows[0]},
                    observation_time=np.array(handoff_times), teacher_phase=np.array(handoff_roles))
        if args.camera_backend == 'csp':
            result.update(pixel_cache_hits=retina.pixel_cache_hits,
                          pixel_cache_misses=retina.pixel_cache_misses)
        write_json(run / 'gc-events.json', gc_events)
        write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
