"""Background teacher-only evaluation, without camera or student training.

Finite speed stages preserve the original source cadence/track/ownership gates.
No lifecycle/UI commands are sent; a stationary forward-gear start is required.
"""
import argparse
import gc
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_latest_tcp import LatestTcpTelemetryClient
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_clock import NativeGraphicsTimingReader
from flybrain_drive.ac_native_physics import (NativePhysicsReader, decode_static_identity_prefix,
    PHYSICS_LAYOUT, PHYSICS_PREFIX)
from flybrain_drive.ac_published_state import read_published_json, validate_async_status_creation
from flybrain_drive.ac_runtime_gc import defer_automatic_cyclic_gc
from flybrain_drive.ac_sac_teacher import (PrivilegedSACTeacher, TeacherReferenceCurve,
    legacy_request_to_normalized_controls, relative_request)
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.ac_teacher_observation import PrivilegedTeacherObservationBuilder
from flybrain_drive.ac_teacher_pilot import check_pilot_state, validate_calibration
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.async_event_archive import AsyncEventArchive
from flybrain_drive.teacher_curriculum import LapEvidence, EvaluationStateGuard, speed_request
from flybrain_drive.source_native_snapshot import WindowsQpc, decode_source_native, source_age_seconds
from flybrain_drive.owned_source_camera import OwnedSourceCamera
from flybrain_drive.control_guard_metadata_process import ControlGuardMetadataProcess
from scripts.record_ac_wgc import physics_record
from scripts.control_ac_session import resume_owned_camera_for_recording


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config', required=True)
    p.add_argument('--calibration-result', required=True)
    p.add_argument('--teacher-config', default='configs/ac-sac-replay.json')
    p.add_argument('--transport', choices=['file', 'tcp'], default='file')
    p.add_argument('--source-native', action='store_true',
        help='Require producer-bundled same-prefix native/SDK reads and original QPC age')
    p.add_argument('--require-quiescent-file-bridge',action='store_true',
        help='Source must explicitly confirm compatibility file I/O is suspended and idle')
    p.add_argument('--camera-plan', help='Prepared owned camera ready.json; data remains unqualified')
    p.add_argument('--shift-handling', choices=['coast', 'policy'], default='coast')
    p.add_argument('--unlimited-final-stage', action='store_true',
        help='Original SAC pedals in the final stage; explicit hard speed ceiling stays active')
    p.add_argument('--max-gap-m', type=float, choices=[3., 5., 12.], default=3.,
        help='Reference-line deviation bound; actual tyres-out termination always remains active')
    p.add_argument('--steering-request-scale', type=float, default=.6)
    p.add_argument('--targets', default='20,40,60,80,120,160,220,280')
    p.add_argument('--stage-seconds', type=float, default=12)
    p.add_argument('--final-seconds', type=float, default=400)
    p.add_argument('--full-laps', type=int, choices=[1,2], default=2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.require_quiescent_file_bridge and (not a.source_native or a.transport!='tcp'):
        raise ValueError('File I/O fence requires the original source-native TCP path')
    targets = [float(x) for x in a.targets.split(',')]
    if (not targets or not 10 <= targets[0] <= 20 or targets[-1] > 300
            or any(not np.isfinite(x) for x in targets)
            or any(y <= x or y-x > 60 for x, y in zip(targets, targets[1:]))
            or not 2 <= a.stage_seconds <= 30 or not 2 <= a.final_seconds <= 600):
        raise ValueError('Bounded ascending teacher curriculum required')
    axis = json.loads(Path(a.axis_config).read_text(encoding='utf8'))
    codec = json.loads(Path(a.teacher_config).read_text(encoding='utf8'))
    geo = json.loads(Path('configs/ac-teacher-geometry-result.json').read_text())['geometry_configuration']
    calibration = json.loads(Path(a.calibration_result).read_text(encoding='utf8'))
    torch.set_num_threads(1)
    run = begin_run(a.output, dict(arguments=vars(a), seed=codec['seed'],
        teacher_configuration=codec, axis_configuration=axis, camera_enabled=bool(a.camera_plan),
        training_admissible=False, student_optimizer_updates=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    spool = Path(os.environ['LOCALAPPDATA'])/'flybrain/event-spool'/uuid.uuid4().hex
    spool.mkdir(parents=True)
    events_path = spool/'events.jsonl'
    result = dict(status='not_ready', teacher_requests=0, feedback=0, stages=[], laps=[],
        training_admissible=False, student_optimizer_updates=0, ui_actions_sent=0,
        official_valid_lap_verified=False, camera_enabled=bool(a.camera_plan))
    session = reader = status_reader = graphics = client = archive = None
    tracker = LapEvidence()
    state_guard = EvaluationStateGuard(max_gap_m=a.max_gap_m)
    native_state = {}
    last_input_receipt = [None]
    last_source_packet = [None]
    source_qpc = WindowsQpc() if a.source_native else None
    camera_owner = None
    metadata_guard = None
    try:
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine required')
        window = windows[0]
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        if identity['car_model'] != 'bmw_z4_gt3' or identity['track'] != 'monza':
            raise ValueError('BMW Z4 GT3/Monza required')
        if sha256(axis['profile_path']) != axis['profile_sha256']:
            raise ValueError('Controls profile differs')
        if sha256(codec['checkpoint']) != codec['checkpoint_sha256']:
            raise ValueError('Teacher checkpoint differs')
        controllers = inspect_vjoy_controllers()
        if (controllers['candidate_ac_joy_index'] != 0 or len(controllers['vjoy_devices']) != 1
                or controllers['vjoy_devices'][0]['instance_guid'] != axis['instance_guid']):
            raise ValueError('DirectInput identity/index differs')
        geometry = PrivilegedTrackGeometry(geo['track_csv'], geo['reference_csv'],
            track_sha256=geo['track_sha256'], reference_sha256=geo['reference_sha256'])
        curve = TeacherReferenceCurve(geo['reference_csv'])
        def new_builder():
            return PrivilegedTeacherObservationBuilder(geometry, curve, profile=codec['profile'],
                minimum_interval=.02, maximum_interval=.08)
        builder = new_builder()
        # Reserve 10ms of the unchanged 50ms source-to-control budget. A 30ms
        # prefilter rejected a real 30.5ms source during a 68ms engine-frame hitch,
        # needlessly skipping the only packet inside the unchanged 80ms cadence.
        client = (LatestTcpTelemetryClient(timeout=.15, clock=time.perf_counter,
            source_admission=(lambda packet: source_age_seconds(packet, source_qpc) <= .04
                and (not a.require_quiescent_file_bridge or
                    packet.raw.get('publisher_file_compatibility_suspended') is True
                    and packet.raw.get('publisher_file_io_quiescent') is True))
                if source_qpc is not None else None)
            if a.transport == 'tcp' else FileTelemetryClient(
                Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry', timeout=.15,
                clock=time.perf_counter))
        reader = NativePhysicsReader(timeout=.15, clock=time.perf_counter)
        status_reader = NativeTeacherStatusReader()
        graphics = NativeGraphicsTimingReader()
        heartbeat_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        before, _ = read_published_json(heartbeat_path)
        if a.camera_plan:
            if source_qpc is None:
                raise ValueError('Camera-labelled collection requires original source-native timing')
            camera_owner=OwnedSourceCamera(a.camera_plan,generation=before['generation'],
                window_handle=window.hwnd,installed_source=Path(window.executable).parent/
                    'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')
            write_json(run/'camera-ownership-plan.json',camera_owner.plan)
        write_json(run/'initial-session.json', before)
        write_json(run/'calibration-input.json', calibration)
        teacher = PrivilegedSACTeacher(codec['checkpoint'], codec['checkpoint_sha256'])
        teacher.predict(np.zeros((1, 125), dtype=np.float32))  # warm inference before acquisition
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        session = VJoySession(sdk, axis['device_id'], [AxisBinding(**b) for b in axis['bindings']])
        metadata_guard=ControlGuardMetadataProcess(heartbeat_path=heartbeat_path,profile_path=axis['profile_path'],
            stop_path=run/'stop',camera_health_path=camera_owner.plan['health_file'] if camera_owner else None)
        if camera_owner is not None:
            if not before['paused']:
                raise ValueError('Camera demonstration requires paused stationary GPU warmup')
            held=reader.receive()
            held_values=iter(PHYSICS_PREFIX.unpack(held.prefix_bytes))
            held_raw={name:tuple(next(held_values) for _ in range(n)) for name,_,n in PHYSICS_LAYOUT}
            if (held_raw['gear'][0]!=2 or held_raw['autoShifterOn'][0]!=1
                    or not all(75<=t<=85 for t in held.fields['tyreCoreTemperature'])
                    or abs(held.fields['speedKmh'])>=.1 or held.fields['gas']>.02
                    or held.fields['brake']<.98):
                raise ValueError('Observed warm 75..85C first-gear full-brake demonstration start required')
            camera_owner.start_paused_warmup()
            before=resume_owned_camera_for_recording(camera_owner,output=run/'owned-camera-resume',
                window=window,axis=axis,sdk=sdk)
        metadata_guard.wait_for_live(before['generation'],before['epoch'],camera_owner.plan['request_id'] if camera_owner else None)
        with AsyncEventArchive(events_path) as archive:
            def emit(kind, **fields):
                archive.emit(dict(kind=kind, host_time=time.perf_counter(), **fields))
            def guard():
                started=time.perf_counter();previous=started;phases={};metadata=None
                def phase(name):
                    nonlocal previous
                    now=time.perf_counter();phases[name]=(now-previous)*1000;previous=now
                try:
                    archive.check();metadata=metadata_guard.snapshot()
                    phase('archive_and_stop_metadata')
                    if api.user.GetForegroundWindow() == window.hwnd:
                        raise RuntimeError('Background engine required')
                    if api.probe(window.hwnd, require_foreground=False) != window:
                        raise RuntimeError('Engine window changed')
                    phase('window_identity')
                    if metadata['profile_sha256'] != axis['profile_sha256']:
                        raise RuntimeError('Controls profile changed')
                    phase('profile_hash')
                    hb, modified = metadata['session'],metadata['session_mtime']
                    validate_async_status_creation(hb)
                    if camera_owner is not None:
                        camera_owner.validate_health(metadata['camera_health'],metadata['camera_health_mtime'],
                            hb['camera_active'],hb.get('camera_request_id'))
                    if (not 0 <= time.time()-modified <= 2 or hb['generation'] != before['generation']
                            or hb['epoch'] != before['epoch'] or hb['paused'] or hb['is_online']
                            or hb['ai_controlled'] or hb['camera_active'] and camera_owner is None
                            or hb['expert_probe_active']
                            or hb.get('error') or hb.get('pending_receipt_id')):
                        raise RuntimeError('Fresh live AI/camera/probe-off original session required')
                    phase('original_session_camera_metadata')
                    flag = status_reader.snapshot()
                    if flag['builtin_ai_controlled']:
                        raise RuntimeError('Native AI owns controls')
                    phase('native_ai')
                finally:
                    emit('guard_timing',duration_ms=(time.perf_counter()-started)*1000,
                        phases_ms=phases,metadata_sequence=metadata['sequence'] if metadata else None,
                        original_metadata_read_started=metadata['read_started'] if metadata else None,unfinished_phase_ms=(time.perf_counter()-previous)*1000)
            def receive():
                # Latest-file publication is asynchronous to this control loop.
                # A new packet can still be too close to the last admitted source
                # time. Wait for a real later packet, logging every skipped pair;
                # never re-date it, reset history or admit a source gap over 80ms.
                admission_deadline = time.perf_counter()+.08
                while True:
                    packet = (client.receive(minimum_source_interval=.02,
                        maximum_source_interval=.08 if builder.previous is not None else None)
                        if isinstance(client, LatestTcpTelemetryClient) else client.receive())
                    control_native = reader.receive()
                    native = (decode_source_native(packet, source_qpc)
                        if source_qpc is not None else control_native)
                    if source_qpc is not None:
                        fields = iter(PHYSICS_PREFIX.unpack(control_native.prefix_bytes))
                        control_fields = {name:tuple(next(fields) for _ in range(n))
                            for name, _, n in PHYSICS_LAYOUT}
                        if control_fields['numberOfTyresOut'][0] > 2 or control_fields['gear'][0] < 1:
                            raise ValueError('Current independent native track exit/reverse')
                    source_delta = (packet.source_monotonic_seconds-builder.previous[3]
                        if builder.previous is not None else None)
                    if source_delta is None or source_delta >= .02:
                        break
                    emit('early_source_pair', sdk=record_packet(packet), native=physics_record(native),
                        source_interval_seconds=source_delta, used_for_policy=False)
                    values = iter(PHYSICS_PREFIX.unpack(native.prefix_bytes))
                    early_fields = {name:tuple(next(values) for _ in range(width))
                        for name, _, width in PHYSICS_LAYOUT}
                    tracker.clean &= early_fields['numberOfTyresOut'][0] == 0
                    if early_fields['numberOfTyresOut'][0] > 2 or early_fields['gear'][0] < 1:
                        raise ValueError('Track exit/reverse observed in early source pair')
                    if source_delta <= 0 or time.perf_counter() >= admission_deadline:
                        raise ValueError('Nonadvancing/too-early source admission deadline')
                    guard()
                timing = graphics.snapshot()
                emit('raw_pair', sdk=record_packet(packet), native=physics_record(native),
                    graphics=timing.metadata(),
                    control_native=physics_record(control_native) if source_qpc is not None else None,
                    source_capture_age_ms=source_age_seconds(packet, source_qpc)*1000
                        if source_qpc is not None else None)
                values = iter(PHYSICS_PREFIX.unpack(native.prefix_bytes))
                native_state.clear()
                for name, _, width in PHYSICS_LAYOUT:
                    field = tuple(next(values) for _ in range(width))
                    native_state[name] = field[0] if width == 1 else field
                current = builder.build(packet, native)
                last_input_receipt[0] = min(packet.received_monotonic_seconds,
                    native.received_monotonic_seconds)
                last_source_packet[0] = packet
                tracker.update(current.state['NormalizedSplinePosition'], timing,
                    current.state['numberOfTyresOut'])
                if camera_owner is not None and session.owned:
                    camera_owner.trigger_near_initial_finish(
                        current.state['NormalizedSplinePosition'],tracker.crossings)
                return current
            guard()
            for _ in range(4):
                started = time.perf_counter()
                current = receive()
                time.sleep(max(0, .04-(time.perf_counter()-started)))
            check_pilot_state(current, require_start=True, max_speed_kmh=40)
            validate_calibration(calibration, axis, current.producer_session_id,
                steering_request_scale=a.steering_request_scale)
            result['preflight_passed'] = True
            if not a.execute:
                result['status'] = 'preflight_only'
            else:
                try:
                    emit('open', request=session.open())
                    builder = new_builder()
                    emit('teacher_history_reset', reason='neutral_device_acquisition')
                    gc.collect()
                    with defer_automatic_cyclic_gc():
                        for _ in range(4):
                            time.sleep(.04)
                            guard()
                            current = receive()
                            check_pilot_state(current, max_speed_kmh=.5)
                        state = current.state
                        if abs(state['steerAngle']) > .5 or state['accStatus'] > .02 or state['brakeStatus'] < .98:
                            raise ValueError('Measured neutral hold required')
                        previous = np.array([0., -1., 1.])
                        next_decision_deadline = time.perf_counter()
                        state_guard.check(current.state, native_state['gear'],
                            native_state['autoShifterOn'], time.perf_counter(), 40)
                        for stage, target in enumerate(targets):
                            start = time.perf_counter()
                            duration = a.final_seconds if stage == len(targets)-1 else a.stage_seconds
                            evidence = dict(target_kmh=target, hard_ceiling_kmh=target+20,
                                duration_budget_seconds=duration, feedback=0, max_speed_kmh=0.)
                            result['stages'].append(evidence)
                            emit('stage_start', **evidence)
                            while time.perf_counter()-start < duration:
                                tick = time.perf_counter()
                                guard()
                                shifting = state_guard.check(current.state, native_state['gear'],
                                    native_state['autoShifterOn'], tick, target+20)
                                if current.observation is None:
                                    raise ValueError('Real teacher history unavailable')
                                if shifting and a.shift_handling == 'coast':
                                    delta = np.zeros(3)
                                    requested = np.array([previous[0], -1., -1.])
                                    limited = requested.copy()
                                else:
                                    delta = teacher.predict(current.observation[None])['deterministic_policy_action'][0]
                                    requested = relative_request(delta, previous, codec['legacy_control_max_delta'],
                                        [codec['legacy_control_bounds'][0]]*3, [codec['legacy_control_bounds'][1]]*3,
                                        product_dtype=codec['legacy_product_dtype'])
                                    limited = (requested.copy() if a.unlimited_final_stage and
                                        stage == len(targets)-1 else speed_request(
                                            requested, current.state['speed']*3.6, target))
                                controls = legacy_request_to_normalized_controls(limited)
                                controls[0] *= a.steering_request_scale
                                inference_finished=time.perf_counter()
                                guard()
                                emit('decision_timing',decision_started=tick,inference_finished=inference_finished,
                                    guard_finished=time.perf_counter(),source_age_ms=source_age_seconds(last_source_packet[0], source_qpc)*1000 if source_qpc else None)
                                if not 0 <= time.perf_counter()-last_input_receipt[0] <= .05:
                                    raise TimeoutError('Original decision inputs exceeded 50ms before actuation')
                                action_source_age = (source_age_seconds(last_source_packet[0], source_qpc)
                                    if source_qpc is not None else None)
                                if action_source_age is not None and action_source_age > .05:
                                    raise TimeoutError(f'Original source capture age {action_source_age*1000:.3f}ms exceeded 50ms before actuation')
                                emit('teacher_request', stage=stage, target_kmh=target,
                                    decision_started_host_time=tick,
                                    next_decision_deadline_host_time=next_decision_deadline+.04,
                                    automatic_shift_coast=bool(shifting and a.shift_handling == 'coast'),
                                    automatic_shift_observed=shifting,
                                    source_capture_age_before_actuation_ms=action_source_age*1000
                                        if action_source_age is not None else None,
                                    observation=current.observation.tolist(), delta=delta.tolist(),
                                    unlimited=requested.tolist(), limited=limited.tolist(),
                                    request=session.submit(*map(float, controls)))
                                result['teacher_requests'] += 1
                                previous = limited
                                next_decision_deadline += .04
                                remaining = next_decision_deadline-time.perf_counter()
                                if remaining <= 0:
                                    raise TimeoutError('Teacher request exceeded 40ms decision budget')
                                time.sleep(remaining)
                                guard()
                                current = receive()
                                speed = current.state['speed']*3.6
                                evidence['feedback'] += 1
                                evidence['max_speed_kmh'] = max(evidence['max_speed_kmh'], speed)
                                result['feedback'] += 1
                                emit('feedback', stage=stage, speed_kmh=speed, gap_m=current.state['gap'],
                                    tyres_out=current.state['numberOfTyresOut'], progress=current.state['NormalizedSplinePosition'])
                                state_guard.check(current.state, native_state['gear'],
                                    native_state['autoShifterOn'], time.perf_counter(), target+20)
                                if len(tracker.laps) >= a.full_laps:
                                    break
                            evidence['completed'] = True
                            emit('stage_end', **evidence)
                            if len(tracker.laps) >= a.full_laps:
                                break
                    result['status'] = 'curriculum_completed'
                finally:
                    if session.owned:
                        result['closing_request'] = session.close()
                        emit('close', request=result['closing_request'])
    except Exception as error:
        result.update(status='aborted', error=type(error).__name__, reason=str(error))
    except BaseException as error:
        result.update(status='interrupted', error=type(error).__name__, reason=str(error))
        raise
    finally:
        if session and session.owned:
            result['emergency_close'] = session.close()
        if metadata_guard is not None:
            result['control_metadata_process']=metadata_guard.close()
            with (run/'guard-metadata.jsonl').open('xb') as metadata_file:
                for payload in metadata_guard.records:metadata_file.write(payload+b'\n')
            result['guard_metadata_sha256']=sha256(run/'guard-metadata.jsonl')
            if (result['control_metadata_process']['status']!='completed' or result['control_metadata_process']['forced_termination']
                    or result['control_metadata_process']['receiver_alive']):
                result.update(status='aborted',reason='Original guard metadata source did not close cleanly')
        if camera_owner is not None:
            camera_owner.request_stop()
            result['owned_camera_stop_requested']=True
        result['actuator_messages_sent'] = session.submit_attempts if session else 0
        if archive is not None:
            result['event_archive'] = archive.statistics()
        result['laps'] = tracker.laps
        result['finish_crossings'] = tracker.crossings
        result['delayed_counter_finishes'] = tracker.delayed_counter_finishes
        for resource in (reader, status_reader, graphics):
            if resource:
                resource.close()
        if isinstance(client, LatestTcpTelemetryClient):
            client.close()
            with (run/'tcp-source-packets.jsonl').open('x', encoding='utf8') as archive:
                for packet in client.packets:
                    archive.write(json.dumps(record_packet(packet), allow_nan=False)+'\n')
            result.update(tcp_source_packets=len(client.packets),
                tcp_source_archive_sha256=sha256(run/'tcp-source-packets.jsonl'),
                tcp_receiver_terminated=not client.thread.is_alive(),
                tcp_cadence_constrained_selections=client.cadence_constrained_selections,
                tcp_collector_error=str(client.error) if client.error else None)
        if events_path.exists():
            shutil.copy2(events_path, run/'events.jsonl')
            if sha256(events_path) != sha256(run/'events.jsonl'):
                raise ValueError('Event archive hash differs')
            result['events_sha256'] = sha256(events_path)
            result['local_spool_retained'] = str(events_path)
        write_json(run/'result.json', result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
