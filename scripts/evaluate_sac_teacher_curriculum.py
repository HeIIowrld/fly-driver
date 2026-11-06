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

from flybrain_drive.ac_geometric_teacher import GeometricRecoveryTeacher
from flybrain_drive.teacher_terminal_stop import lateral_request,run_terminal_stop
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
    p.add_argument('--low-speed-lateral-calibration',help='Explicit historical teacher-only curvature fit; includes verified terminal stop')
    p.add_argument('--axis-config', required=True)
    p.add_argument('--calibration-result', required=True)
    p.add_argument('--teacher-config', default='configs/ac-sac-replay.json')
    p.add_argument('--transport', choices=['file', 'tcp'], default='file')
    p.add_argument('--source-native', action='store_true',
        help='Require producer-bundled same-prefix native/SDK reads and original QPC age')
    p.add_argument('--require-quiescent-file-bridge',action='store_true',
        help='Source must explicitly confirm compatibility file I/O is suspended and idle')
    p.add_argument('--camera-plan', help='Prepared owned camera ready.json; data remains unqualified')
    p.add_argument('--thermal-initialization-run', help='Audited experimental thermal condition; requires owned camera and current same-session temperature')
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
    p.add_argument('--heldout-geometric-speed', action='store_true',
        help='Use the teacher-only geometric steering and speed policy together')
    p.add_argument('--heldout-magione-geometry', type=Path,
        help='Separate bounded teacher-only Magione calibration with verified geometry')
    p.add_argument('--magione-stage-audit', help='Exact independently audited stationary stage for the owned student')
    p.add_argument('--moving-stop-reference', help='Explicit teacher-only emergency stop calibration')
    p.add_argument('--moving-student-plan', help='Owned student mailbox plan for a calibrated moving trial')
    p.add_argument('--moving-stop-audit', help='Independent stop audit for the same moving reference')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.thermal_initialization_run and not a.camera_plan:
        raise ValueError('Thermal condition collection requires owned synchronized camera')
    if a.require_quiescent_file_bridge and (not a.source_native or a.transport!='tcp'):
        raise ValueError('File I/O fence requires the original source-native TCP path')
    targets = [float(x) for x in a.targets.split(',')]
    if (not targets or not 10 <= targets[0] <= 20 or targets[-1] > 300
            or any(not np.isfinite(x) for x in targets)
            or any(y <= x or y-x > 60 for x, y in zip(targets, targets[1:]))
            or not 2 <= a.stage_seconds <= 30 or not 2 <= a.final_seconds <= 600):
        raise ValueError('Bounded ascending teacher curriculum required')
    if not a.heldout_magione_geometry and a.low_speed_lateral_calibration and (not a.source_native or targets!=[10.,20.] or a.unlimited_final_stage or a.stage_seconds!=2 or a.final_seconds!=2):
        raise ValueError('Explicit finite10/20kmh two-second lateral/stop diagnostic only')
    heldout = a.heldout_magione_geometry is not None
    expected_track = 'magione' if heldout else 'monza'
    heldout_lateral = heldout and bool(a.low_speed_lateral_calibration)
    if a.heldout_geometric_speed and not heldout_lateral:
        raise ValueError('Geometric speed requires the separate Magione teacher experiment')
    heldout_student = heldout and a.moving_student_plan is not None
    if bool(a.magione_stage_audit) != heldout_student:
        raise ValueError('An audited Magione stage is required only for a staged Magione student trial')
    if heldout_student and (not a.heldout_geometric_speed or not a.camera_plan
            or a.thermal_initialization_run or not a.require_quiescent_file_bridge):
        raise ValueError('Staged Magione requires the geometric teacher and owned source-native camera')
    heldout_seconds = 55 if heldout_student else 230 if a.heldout_geometric_speed else 170 if heldout_lateral else 90
    heldout_profile = ('magione_close_corner_30_v1' if heldout_lateral
                       else 'magione_close_corner_60_v1')
    heldout_targets = [20., 25., 30.] if heldout_lateral else [20., 35., 50., 60.]
    if heldout and (a.transport != 'tcp' or not a.source_native or not a.moving_stop_reference
            or (a.camera_plan and not heldout_student) or a.unlimited_final_stage
            or targets != heldout_targets or a.stage_seconds != 3
            or a.final_seconds != heldout_seconds or a.max_gap_m != 3):
        raise ValueError('Exact bounded teacher-only Magione stop calibration required')
    if a.moving_stop_reference and (not a.source_native or not (a.camera_plan or heldout)
            or (a.low_speed_lateral_calibration and not heldout_lateral) or not a.execute or a.full_laps!=1):
        raise ValueError('Moving stop calibration requires the owned source-native teacher recording')
    if a.moving_stop_reference:
        reference = json.loads(Path(a.moving_stop_reference).read_text(encoding='utf-8'))
        reference_profile = reference.get('teacher_pacing_profile')
        from flybrain_drive.magione_stage import MAGIONE_MOVING_STAGE_PROFILES
        accepted_heldout_profiles = {heldout_profile}
        if a.heldout_geometric_speed:
            accepted_heldout_profiles.update(MAGIONE_MOVING_STAGE_PROFILES)
            if not heldout_student:
                accepted_heldout_profiles.update(MAGIONE_MOVING_STAGE_PROFILES.values())
        if (heldout and reference_profile not in accepted_heldout_profiles
                or not heldout and reference_profile in ('magione_close_corner_30_v1', 'magione_alternative_corner_30_v1',
                    'magione_alternative_staging_30_v1',
                    'magione_close_corner_60_v1', 'magione_staging_30_v1')):
            raise ValueError('Moving reference profile and experiment track differ')
        if (reference.get('teacher_pacing_profile')=='monza_straight_120_v1'
                and (targets!=[20.,40.,60.,80.,119. if a.moving_student_plan else 120.]
                     or a.unlimited_final_stage
                     or a.stage_seconds!=3 or a.final_seconds!=55)):
            raise ValueError('Exact capped teacher pacing profile required')
    if bool(a.moving_student_plan)!=bool(a.moving_stop_audit):
        raise ValueError('Moving student plan and independent stop audit must be provided together')
    if a.moving_student_plan:
        if not a.moving_stop_reference:
            raise ValueError('Exact previously calibrated moving reference required')
        from flybrain_drive.moving_flight_trial import MovingStudentTrial, admit_calibration
        admit_calibration(a.moving_stop_reference,a.moving_stop_audit)
        moving_plan = json.loads(Path(a.moving_student_plan).read_text(encoding='utf-8'))
        if (Path(a.moving_student_plan).resolve().parent!=Path(a.camera_plan).resolve().parent
                or Path(moving_plan['terminal_path']).resolve()!=Path(a.camera_plan).resolve().parent/'student-terminal.json'):
            raise ValueError('Student coordination paths must stay inside the owned camera session')
    axis = json.loads(Path(a.axis_config).read_text(encoding='utf8'))
    codec = json.loads(Path(a.teacher_config).read_text(encoding='utf8'))
    geometry_path = a.heldout_magione_geometry or Path('configs/ac-teacher-geometry-result.json')
    geometry_evidence = json.loads(geometry_path.read_text(encoding='utf-8'))
    if heldout and (geometry_evidence.get('status') != 'verified_offline_magione_teacher_geometry'
            or geometry_evidence.get('actor_input') is not False
            or sha256(geometry_path) != reference.get('geometry_result_sha256')
            or axis['track'] != expected_track):
        raise ValueError('Exact teacher-only Magione geometry and axis evidence required')
    geo = geometry_evidence['geometry_configuration']
    calibration = json.loads(Path(a.calibration_result).read_text(encoding='utf8'))
    torch.set_num_threads(1)
    run = begin_run(a.output, dict(arguments={**vars(a), 'heldout_magione_geometry':
        str(a.heldout_magione_geometry) if heldout else None}, seed=codec['seed'],
        expected_track=expected_track, geometry_result_sha256=sha256(geometry_path),
        teacher_configuration=codec, axis_configuration=axis,
        low_speed_lateral_fit_sha256=sha256(a.low_speed_lateral_calibration) if a.low_speed_lateral_calibration else None,
        terminal_stop_module_sha256=sha256('src/flybrain_drive/teacher_terminal_stop.py'), camera_enabled=bool(a.camera_plan),
        moving_stop_reference_sha256=sha256(a.moving_stop_reference) if a.moving_stop_reference else None,
        moving_student_plan_sha256=sha256(a.moving_student_plan) if a.moving_student_plan else None,
        magione_stage_audit_sha256=sha256(a.magione_stage_audit) if a.magione_stage_audit else None,
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
    camera_owner = staging_pause = None
    metadata_guard = None
    position_reader = stop_calibration = latest_control_native = moving_trial = None
    try:
        if (heldout and not heldout_student
                and reference_profile == 'magione_alternative_staging_30_v1'):
            from flybrain_drive.owned_pause_process import OwnedPauseProcess
            from scripts.pilot_mailbox_flight_control import CPU, CPULIB
            from flybrain_drive.sac_unified_student import read
            staging_pause = OwnedPauseProcess(run/'staging-pause',read(run/'manifest.json'),
                reference['installed_lua_sha256'],CPU,CPULIB,heldout_magione=True)
            write_json(run/'staging-pause-ready.json',staging_pause.ready)
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine required')
        window = windows[0]
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        if identity['car_model'] != 'bmw_z4_gt3' or identity['track'] != expected_track:
            raise ValueError('Expected BMW Z4 GT3 and native track required')
        if sha256(axis['profile_path']) != axis['profile_sha256']:
            raise ValueError('Controls profile differs')
        if sha256(codec['checkpoint']) != codec['checkpoint_sha256']:
            raise ValueError('Teacher checkpoint differs')
        controllers = inspect_vjoy_controllers()
        if (controllers['candidate_ac_joy_index'] != 0 or len(controllers['vjoy_devices']) != 1
                or controllers['vjoy_devices'][0]['instance_guid'] != axis['instance_guid']):
            raise ValueError('DirectInput identity/index differs')
        geometry = PrivilegedTrackGeometry(geo['track_csv'], geo['reference_csv'],
            track_sha256=geo['track_sha256'], reference_sha256=geo['reference_sha256'],
            cyclic_reference=heldout)
        lateral=None
        if a.low_speed_lateral_calibration:
            fit_path=Path(a.low_speed_lateral_calibration);fit=json.loads(fit_path.read_text(encoding='utf8'))
            old=json.loads((fit_path.parent/'manifest.json').read_text(encoding='utf8'))['config']['source_configuration']
            old_cal=json.loads(Path(old['arguments']['calibration_result']).read_text(encoding='utf8'))
            if (fit['status']!='completed' or fit['teacher_only'] is not True or old['axis_configuration']['car_model']!=axis['car_model']
                    or old_cal['bindings']!=axis['bindings'] or old_cal['dll_sha256']!=axis['dll_sha256']
                    or abs(old_cal['sdk_raw_per_normalized_request']/calibration['sdk_raw_per_normalized_request']-1)>.02):
                raise ValueError('Historical teacher curvature fit/control response is incompatible')
            lateral=GeometricRecoveryTeacher(geometry.line,fit['curvature_per_canonical_steering'],
                maximum_speed_kmh=30. if heldout_lateral else 20.)
            write_json(run/'teacher-lateral-contract.json',dict(kind='historical_kinematic_teacher_only_transfer',
                fit_path=str(fit_path),fit_sha256=sha256(fit_path),original_profile_sha256=fit['profile_sha256'],
                current_profile_sha256=axis['profile_sha256'],current_sdk_gain=calibration['sdk_raw_per_normalized_request'],
                historical_sdk_gain=old_cal['sdk_raw_per_normalized_request'],current_moving_curvature_verified=False,
                privileged_route=True,actor_input=False,steering_request_scale=a.steering_request_scale))
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
        if a.moving_stop_reference:
            from flybrain_drive.ac_native_position import NativeDiagnosticPositionReader
            from flybrain_drive.moving_stop_calibration import MovingStopCalibration
            position_reader = NativeDiagnosticPositionReader()
            reference = json.loads(Path(a.moving_stop_reference).read_text(encoding='utf-8'))
            stop_calibration = MovingStopCalibration(reference, position_reader)
        heartbeat_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        before, _ = read_published_json(heartbeat_path)
        if heldout:
            installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
            if (before['track_id'] != expected_track or before['car_id'] != identity['car_model']
                    or before.get('capabilities', {}).get('heldout_track_lifecycle_v1') is not True
                    or sha256(installed) != reference.get('installed_lua_sha256')):
                raise ValueError('Exact live Magione lifecycle deployment required')
        if a.camera_plan:
            if source_qpc is None:
                raise ValueError('Camera-labelled collection requires original source-native timing')
            camera_owner=OwnedSourceCamera(a.camera_plan,generation=before['generation'],
                window_handle=window.hwnd,installed_source=Path(window.executable).parent/
                    'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')
            write_json(run/'camera-ownership-plan.json',camera_owner.plan)
        write_json(run/'initial-session.json', before)
        write_json(run/'calibration-input.json', calibration)
        teacher = None
        if not a.heldout_geometric_speed:
            teacher = PrivilegedSACTeacher(codec['checkpoint'], codec['checkpoint_sha256'])
            teacher.predict(np.zeros((1, 125), dtype=np.float32))  # warm before acquisition
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
                    or abs(held.fields['speedKmh'])>=.1 or held.fields['gas']>.02
                    or held.fields['brake']<.98):
                raise ValueError('Observed first-gear autoshifter stationary full-brake demonstration start required')
            magione_stage = None
            if heldout_student:
                from flybrain_drive.magione_stage import admit_magione_stage
                magione_stage = admit_magione_stage(a.magione_stage_audit,before,held,position_reader.snapshot(),
                    installed_sha256=camera_owner.plan['source_sha256'],now=time.perf_counter())
                if (magione_stage['geometry_result_sha256']!=sha256(geometry_path)
                        or magione_stage['staging_profile']!=MAGIONE_MOVING_STAGE_PROFILES[reference_profile]):
                    raise ValueError('Staging and moving Magione route evidence differ')
                write_json(run/'magione-stage-admission.json',magione_stage)
            elif a.thermal_initialization_run:
                from scripts.audit_initial_thermal_evidence import audit
                from flybrain_drive.thermal_teacher_start import admit_thermal_teacher_start
                admission=admit_thermal_teacher_start(audit(a.thermal_initialization_run),before,
                    held.fields['tyreCoreTemperature'],source_sha256=camera_owner.plan['source_sha256'])
                write_json(run/'thermal-start-admission.json',admission)
                result['thermal_start_admission']=admission
            elif not all(75<=t<=85 for t in held.fields['tyreCoreTemperature']):
                raise ValueError('Observed warm 75..85C demonstration start required without explicit thermal evidence')
            camera_owner.start_paused_warmup()
            before=resume_owned_camera_for_recording(camera_owner,output=run/'owned-camera-resume',
                window=window,axis=axis,sdk=sdk,thermal_admission=result.get('thermal_start_admission'),
                magione_stage=magione_stage)
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
                            or hb['track_id'] != expected_track
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
                nonlocal latest_control_native
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
                    latest_control_native = control_native
                    native = (decode_source_native(packet, source_qpc)
                        if source_qpc is not None else control_native)
                    if source_qpc is not None:
                        fields = iter(PHYSICS_PREFIX.unpack(control_native.prefix_bytes))
                        control_fields = {name:tuple(next(fields) for _ in range(n))
                            for name, _, n in PHYSICS_LAYOUT}
                        if (control_fields['numberOfTyresOut'][0] > (0 if heldout else 2)
                                or control_fields['gear'][0] < 1):
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
                    if a.moving_student_plan:
                        moving_trial=MovingStudentTrial(plan=moving_plan,reference_path=a.moving_stop_reference,
                            audit_path=a.moving_stop_audit,generation=before['generation'],epoch=before['epoch'],
                            request_id=camera_owner.plan['request_id'],calibration=stop_calibration,
                            session=session,guard=guard,emit=emit)
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
                            ceiling = min(target+20, 40) if heldout_lateral else target+20
                            evidence = dict(target_kmh=target, hard_ceiling_kmh=ceiling,
                                duration_budget_seconds=duration, feedback=0, max_speed_kmh=0.)
                            result['stages'].append(evidence)
                            emit('stage_start', **evidence)
                            while time.perf_counter()-start < duration:
                                tick = time.perf_counter()
                                guard()
                                shifting = state_guard.check(current.state, native_state['gear'],
                                    native_state['autoShifterOn'], tick, ceiling)
                                gate = None
                                if stop_calibration is not None:
                                    gate = stop_calibration.observe(last_source_packet[0], latest_control_native,
                                        current.state['NormalizedSplinePosition'], emit)
                                    if gate is not None and moving_trial is None:
                                        result['moving_stop_calibration'] = stop_calibration.stop(
                                            gate, session=session, reader=reader, guard=guard, emit=emit)
                                        break
                                if current.observation is None:
                                    raise ValueError('Real teacher history unavailable')
                                if shifting and a.shift_handling == 'coast':
                                    delta = np.zeros(3)
                                    requested = np.array([previous[0], -1., -1.])
                                    limited = requested.copy()
                                elif a.heldout_geometric_speed:
                                    delta = None
                                    requested = lateral.request(current.state)
                                    pacing_target = target
                                    if heldout_student:
                                        progress = current.state['NormalizedSplinePosition']
                                        if reference_profile == 'magione_alternative_corner_30_v1' and progress >= .061:
                                            pacing_target = min(target,moving_trial.maximum_calibrated_speed_kmh-4.9)
                                        elif reference_profile == 'magione_close_corner_30_v1' and progress >= .298:
                                            pacing_target = min(target,moving_trial.maximum_calibrated_speed_kmh-.55)
                                    limited = speed_request(requested,current.state['speed']*3.6,pacing_target)
                                else:
                                    delta = teacher.predict(current.observation[None])['deterministic_policy_action'][0]
                                    requested = relative_request(delta, previous, codec['legacy_control_max_delta'],
                                        [codec['legacy_control_bounds'][0]]*3, [codec['legacy_control_bounds'][1]]*3,
                                        product_dtype=codec['legacy_product_dtype'])
                                    limited = (requested.copy() if a.unlimited_final_stage and
                                        stage == len(targets)-1 else speed_request(
                                            requested, current.state['speed']*3.6, target))
                                if lateral is not None and not a.heldout_geometric_speed:
                                    limited=lateral_request(limited,current.state,lateral)
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
                                if moving_trial is None:
                                    receipt=session.submit(*map(float, controls))
                                else:
                                    original=last_source_packet[0].raw['source_native_snapshot']
                                    source_receipt=original['qpc_started_ticks']/source_qpc.frequency
                                    receipt=moving_trial.pacing(controls,result['teacher_requests']+1,source_receipt)
                                emit('teacher_request', stage=stage, target_kmh=target,
                                    decision_started_host_time=tick,
                                    next_decision_deadline_host_time=next_decision_deadline+.04,
                                    automatic_shift_coast=bool(shifting and a.shift_handling == 'coast'),
                                    automatic_shift_observed=shifting,
                                    source_capture_age_before_actuation_ms=action_source_age*1000
                                        if action_source_age is not None else None,
                                    teacher_lateral_diagnostics=lateral.last_diagnostics if lateral is not None else None,
                                    teacher_policy='geometric' if a.heldout_geometric_speed else 'original_sac',
                                    observation=current.observation.tolist(),
                                    delta=None if delta is None else delta.tolist(),
                                    unlimited=requested.tolist(), limited=limited.tolist(),
                                    request=receipt)
                                result['teacher_requests'] += 1
                                if gate is not None:
                                    result['moving_student_trial']=moving_trial.run(gate,reader=reader)
                                    break
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
                                    native_state['autoShifterOn'], time.perf_counter(), ceiling)
                                if len(tracker.laps) >= a.full_laps:
                                    break
                            evidence['completed'] = True
                            emit('stage_end', **evidence)
                            if ('moving_stop_calibration' in result or 'moving_student_trial' in result
                                    or len(tracker.laps) >= a.full_laps):
                                break
                        if lateral is not None and not heldout:
                            def stopping_source_guard():
                                if not 0<=time.perf_counter()-last_input_receipt[0]<=.05:raise TimeoutError('Stopping inputs exceed50ms')
                                age=source_age_seconds(last_source_packet[0],source_qpc)
                                if not 0<=age<=.05:raise TimeoutError('Stopping original source capture exceeds50ms')
                                return age
                            result['terminal_stop']=run_terminal_stop(session,current,receive,lateral,guard,stopping_source_guard,emit,
                                steering_request_scale=a.steering_request_scale,
                                state_guard=lambda current:state_guard.check(current.state,native_state['gear'],native_state['autoShifterOn'],time.perf_counter(),40))
                    if moving_trial is not None:
                        if result.get('moving_student_trial',{}).get('status')=='completed_moving_interval':
                            result['status']='completed_moving_student_trial'
                        else:
                            result.update(status='aborted',reason='Calibrated student interval not verified')
                    elif stop_calibration is None:
                        result['status'] = 'curriculum_completed'
                    elif result.get('moving_stop_calibration', {}).get('status') == 'completed_stop_measurement':
                        result['status'] = 'teacher_moving_stop_calibration_completed'
                    else:
                        result.update(status='aborted', reason='Moving stop gate or track-limited stop was not verified')
                finally:
                    if session.owned:
                        if moving_trial is not None and not result.get('moving_student_trial',{}).get('cleanup',{}).get('stationary_verified'):
                            try:
                                result['moving_failure_stop']=moving_trial.hold_failure('pacing_or_student_failed')
                            except Exception as error:
                                result['moving_failure_stop']=dict(stationary_verified=False,error=str(error))
                        if heldout and not result.get('moving_stop_calibration', {}).get('stationary_verified'):
                            try:
                                from flybrain_drive.moving_stop_calibration import native_track_fields
                                hold = session.submit(0., 0., 1.)
                                emit('heldout_failure_stop_request', request=hold)
                                if not hold['driver_accepted']:
                                    raise RuntimeError('Heldout failure brake was not accepted')
                                deadline = time.perf_counter()+8
                                stationary = maximum_tyres = 0
                                damage = native_track_fields(latest_control_native)[1]
                                new_damage = False
                                while time.perf_counter() < deadline:
                                    sample = reader.receive()
                                    if status_reader.snapshot()['builtin_ai_controlled']:
                                        raise ValueError('Native AI changed during heldout stop')
                                    tyre, current_damage = native_track_fields(sample)
                                    maximum_tyres = max(maximum_tyres, tyre)
                                    new_damage |= any(x > y+.01 for x,y in zip(current_damage,damage))
                                    fields = sample.fields
                                    held = abs(fields['speedKmh']) < .1 and fields['gas'] < .02 and fields['brake'] >= .98
                                    stationary = stationary+1 if held else 0
                                    emit('heldout_failure_stop_native', native=physics_record(sample), tyres_out=tyre)
                                    if stationary >= 3:
                                        break
                                result['heldout_failure_stop'] = dict(stationary_verified=stationary >= 3,
                                    maximum_tyres_out=maximum_tyres, new_damage=new_damage)
                            except Exception as error:
                                result['heldout_failure_stop'] = dict(stationary_verified=False,error=str(error))
                        result['closing_request'] = session.close()
                        emit('close', request=result['closing_request'])
                        if (staging_pause is not None
                                and (result.get('moving_stop_calibration', {}).get('stationary_verified')
                                     or result.get('heldout_failure_stop', {}).get('stationary_verified'))):
                            result['staging_pause_action'] = staging_pause.pause()
    except Exception as error:
        result.update(status='aborted', error=type(error).__name__, reason=str(error))
    except BaseException as error:
        result.update(status='interrupted', error=type(error).__name__, reason=str(error))
        raise
    finally:
        if session and session.owned:
            result['emergency_close'] = session.close()
        if staging_pause is not None:
            try:
                result['staging_pause_terminal'] = staging_pause.close()
                if staging_pause.called and result['staging_pause_terminal']['exitcode'] != 0:
                    raise RuntimeError('Original staging pause worker did not exit cleanly')
            except Exception as error:
                result.update(status='aborted', staging_pause_error=str(error))
        if moving_trial is not None:
            result['student_requests']=sum(e['kind']=='student_request' and
                e.get('receipt',{}).get('driver_accepted') is True for e in moving_trial.lease.events)
            moving_trial.close()
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
        for resource in (reader, status_reader, graphics, position_reader):
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
    if heldout:
        expected_status = 'completed_moving_student_trial' if heldout_student else 'teacher_moving_stop_calibration_completed'
        if result['status'] != expected_status:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
