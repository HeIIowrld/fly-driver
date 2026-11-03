# ruff: noqa: E501 -- Exact source-derived nested context is intentional.
"""Finite owned engine-AI motion with four-view/body recording, no desktop input."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision_lifecycle_status_v1 import CSPVisionRecorder, NoFreshCSPFrame
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import (
    NativeTeacherPhysicsReader,
    TeacherPhysicsSample,
    decode_teacher_physics,
)
from flybrain_drive.actor_cadence_trace_v1 import ActorTrace, load_actor_profile
from flybrain_drive.camera_status_policy_v1 import load_profile
from flybrain_drive.collection_cleanup_v4 import recorder_cleanup
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.frame_persistence_profile_v4 import (
    HostFrameWriterFactory,
    load_host_copy_profile,
)
from flybrain_drive.graph import sha256
from flybrain_drive.teacher_pace import PACE_COMMAND, PACE_CONTRACT, goal_matches
from scripts.record_ac_wgc import physics_record

DEFAULT_RETINA_CONFIG = 'configs/ac-csp-yaw-retina.json'



SAME_COPY_SOURCE_FILES = (
    'src/flybrain_drive/ac_teacher_physics.py',
    'src/flybrain_drive/ac_native_physics.py',
    'src/flybrain_drive/ac_teacher_status.py',
    'src/flybrain_drive/ac_image_buffer.py',
    'src/flybrain_drive/ac_csp_vision_lifecycle_status_v1.py',
    'src/flybrain_drive/camera_status_policy_v1.py',
    'integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua',
    'src/flybrain_drive/ac_csp_retina.py',
    'scripts/record_ac_wgc.py',
    'src/flybrain_drive/frame_persistence_profile_v4.py',
    'src/flybrain_drive/ac_shared_texture_host_v4.py',
    'src/flybrain_drive/collection_cleanup_v4.py',
    'src/flybrain_drive/actor_cadence_trace_v1.py',
)


def preserve_native_authority_sources(run):
    """Opt-in proof before camera/control activation; no payload or live access."""
    original = Path(__file__).read_bytes()
    with (run/'executed-program.py').open('xb') as stream:
        stream.write(original)
    records = {}
    for name in SAME_COPY_SOURCE_FILES:
        data = Path(name).read_bytes()
        records[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    proof = dict(schema='flybrain.native.same_copy_recorder_source_proof.v1',
        executed_program_sha256=hashlib.sha256(original).hexdigest(),
        source_files=records, before_camera_activation=True, before_control_activation=True,
        actor_fields_changed=False, fit_permission_granted=False)
    write_json(run/'native-authority-source-proof.json', proof)
    return proof


def native_recording_reader(same_copy_native_authority, *, clock):
    """Same original deadline, poll cadence and vision host receipt clock."""
    if same_copy_native_authority:
        return NativeTeacherPhysicsReader(timeout=.35, poll_seconds=.005, clock=clock)
    return NativePhysicsReader(timeout=.35, clock=clock)


def receive_recording_native(reader, *, same_copy_native_authority):
    sample = reader.receive()
    if not same_copy_native_authority:
        return sample, None
    if not isinstance(sample, TeacherPhysicsSample):
        raise ValueError('Same-copy mode requires a coupled420-byte reader result')
    # Re-parse the saved bytes, without another mapping read or clock sample.
    # A flag/packet/receipt from a separate snapshot cannot masquerade as coupled.
    verified = decode_teacher_physics(sample.authority_prefix_bytes,
                                     sample.physics.received_monotonic_seconds)
    if type(sample.builtin_ai_controlled) is not bool or sample != verified:
        raise ValueError('Same-copy body, native flag or original receipt differs')
    return sample.physics, sample


def native_authority_metadata(coupled, *, baseline_packet, decision_time,
                              csp_running, maximum_body_age, teacher_kind):
    """Necessary native-flag evidence only; never a fit or camera-label grant."""
    if (type(csp_running) is not bool or type(decision_time) not in (int, float)
            or type(maximum_body_age) not in (int, float)
            or not math.isfinite(decision_time) or decision_time < 0
            or not math.isfinite(maximum_body_age) or maximum_body_age <= 0):
        raise ValueError('Finite original decision clock and existing retina bound required')
    record = coupled.authority_record()
    age = decision_time-coupled.physics.received_monotonic_seconds
    checks = dict(native_AI_true=coupled.builtin_ai_controlled is True,
        packet_advanced_since_baseline=coupled.physics.packet_id > baseline_packet,
        CSP_AI_running_at_loop_start=csp_running,
        receipt_age_within_existing_retina_bound=0 <= age <= maximum_body_age)
    record.update(teacher_authority_eligible=all(checks.values()),
        eligibility_scope='same_copy_native_AI_flag_at_body_packet_only',
        eligibility_checks=checks, ineligible_reasons=[k for k, v in checks.items() if not v],
        decision_monotonic_seconds=decision_time, receipt_age_at_decision_seconds=age,
        maximum_body_age_seconds=maximum_body_age, teacher_kind=teacher_kind,
        camera_teacher_authority_synchronized=False, exclusive_teacher_control_verified=False,
        simulation_clock_verified=False, fit_permission_granted=False)
    return record


def verified_shadow(config_file, verification):
    """Only admit the same inference sources/config that passed complete original replay."""
    config = json.loads(Path(config_file).read_text(encoding='utf8'))
    verified = Path(verification)
    receipt = json.loads((verified/'result.json').read_text(encoding='utf8'))
    manifest = json.loads((verified/'manifest.json').read_text(encoding='utf8'))
    if (receipt.get('status') != 'completed_coupled_stream_replay'
            or receipt.get('author_and_flight_parameters_unchanged') is not True
            or manifest['config']['config_sha256'] != sha256(config_file)):
        raise ValueError('Successful exact coupled replay required before live shadow inference')
    for source in ['flight_stream', 'flyvis_camera', 'flyvis_compat', 'virtual_body',
                   'flight_circuit', 'flight_commands', 'flight_navigation', 'vehicle_circuit',
                   'flight_learning', 'ac_csp_vision', 'ac_csp_geometry', 'ac_native_physics',
                   'sensor_contract']:
        file = 'src/flybrain_drive/'+source+'.py'
        if manifest['source_sha256'].get(file) != sha256(file):
            raise ValueError('Verified shadow inference source changed: '+source)
    return config, dict(verification=str(verified),
        result_sha256=sha256(verified/'result.json'), config_sha256=sha256(config_file),
        initialization_seed=manifest['config']['seed'])


def expert_command(duration, speed_hint, recover_to15=False, lateral_recovery=False,
                   brake_pulse=False, pace=False):
    if pace:
        if (recover_to15 or lateral_recovery or brake_pulse
                or (duration, speed_hint) != (30, 100)):
            raise ValueError('Explicit pace task requires separate30sec100kmh maximum preset')
        return PACE_COMMAND
    if brake_pulse:
        if recover_to15 or lateral_recovery or (duration, speed_hint) != (30, 30):
            raise ValueError('Brake pulse requires the separate fixed30sec30kmh preset')
        return 'expert-brake-pulse30'
    if lateral_recovery:
        if recover_to15 or (duration, speed_hint) != (30, 15):
            raise ValueError('Lateral recovery requires the separate fixed30sec15kmh preset')
        return 'expert-lateral-recovery15'
    if recover_to15:
        if duration != 30 or speed_hint not in (20, 30):
            raise ValueError('Recovery requires a fixed30sec warmup20/30-to-target15 preset')
        return 'expert-speed20-recovery15' if speed_hint == 20 else 'expert-speed-recovery15'
    commands = {(3, 15): 'expert-motion-probe', (30, 15): 'expert-collect30',
                (3, 30): 'expert-motion-speed30', (30, 30): 'expert-collect30-speed30',
                (3, 60): 'expert-motion-speed60', (30, 60): 'expert-collect30-speed60'}
    if (duration, speed_hint) not in commands:
        raise ValueError('Only explicit tested finite expert profiles may be requested')
    return commands[duration, speed_hint]


def read_ai_phase(path, generation, *, timeout=.25, task=False, clock=time.monotonic):
    """Retry only a temporarily replaced heartbeat, never a control request."""
    from flybrain_drive.session_heartbeat_provenance import (
        HeartbeatProvenanceError,
        capture_lease,
        validate_opened_status,
    )

    deadline = time.monotonic()+timeout
    while True:
        try:
            captured = capture_lease(path)
            with path.open('rb') as stream:
                content = stream.read()
                value = json.loads(content.decode('utf8'))
                modified = os.fstat(stream.fileno()).st_mtime
                age = time.time()-modified
            validate_opened_status(path, content, value, captured_lease=captured)
            if (not 0 <= age <= 2 or value['generation'] != generation or value.get('error')):
                raise ValueError('Fresh same-generation engine-AI status required')
            if task:
                if value.get('ai_controlled') is not True or value.get('paused') is not False:
                    return None  # No actor observation has been collected in this iteration yet.
                current = value.get('external_task')
                if (not current or current.get('schema') != PACE_CONTRACT
                        or current.get('actor_input') is not True
                        or current.get('phase') not in ('accelerate60', 'accelerate100', 'stop0')
                        or not goal_matches(current.get('desired_speed_m_s'), {
                            'accelerate60': 60/3.6, 'accelerate100': 100/3.6,
                            'stop0': 0}.get(current.get('phase')))):
                    raise ValueError('Explicit externally commanded speed task unavailable: '+
                        json.dumps(dict(external_task=current, source_clock=value['script_clock'])))
                return dict(current, source_script_clock=value['script_clock'],
                    observed_at_host_seconds=clock(), status_file_modified_unix=modified,
                    held_external_instruction=True)
            return value.get('ai_controlled') is True and value.get('paused') is False
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, HeartbeatProvenanceError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.002)


def main(*, heartbeat_start=None, actor_trace_sink=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_RETINA_CONFIG)
    parser.add_argument('--image-format', choices=['png', 'bmp'], default='png')
    parser.add_argument('--shared-transport', action='store_true')
    parser.add_argument('--batched-shared-copy', action='store_true',
                        help='Four staging copies then one Flush, with a bounded group poll')
    parser.add_argument('--async-pixel-precompute', action='store_true',
                        help='Bounded pixel cache warming; exact actor fallback unchanged')
    parser.add_argument('--async-frame-persistence', action='store_true',
                        help='Immutable RAM receipt with bounded deferred BMP persistence')
    parser.add_argument('--mmf-metadata', action='store_true',
                        help='Owned v3 metadata/ACK maps with exact raw slot preservation')
    parser.add_argument('--source-timing', action='store_true',
                        help='Opt-in finite producer call timing; terminal-only bounded ledger')
    parser.add_argument('--same-copy-native-authority', action='store_true',
                        help='Opt-in same420-byte body/privileged AI metadata; no actor field')
    parser.add_argument('--ai-steer-multiplier', type=float, choices=[1., .9], default=1.)
    parser.add_argument('--duration-seconds', type=int, choices=[3, 30], default=3)
    parser.add_argument('--speed-hint-kmh', type=int, choices=[15, 20, 30, 60, 100], default=15)
    parser.add_argument('--recover-to15', action='store_true')
    parser.add_argument('--brake-pulse', action='store_true',
                        help='One engineering .4second brake intervention, not natural AI labels')
    parser.add_argument('--pace-task', action='store_true',
                        help='Explicit60/100/stop/60 task; not natural AI driving labels')
    parser.add_argument('--lateral-recovery', action='store_true')
    parser.add_argument('--local-event-spool', action='store_true',
                        help='Retain original events locally, then copy and hash-check on NAS')
    parser.add_argument('--flight-shadow-config',
                        help='Optional coupled GPU inference, never connected to actuators')
    parser.add_argument('--flight-shadow-verification',
                        help='Successful exact source/config replay required for live shadow mode')
    parser.add_argument('--frame-persistence-profile', required=True)
    parser.add_argument('--actor-cadence-profile', required=True)
    parser.add_argument('--camera-status-profile', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    profile = load_host_copy_profile(args.frame_persistence_profile)
    if (not all((args.async_frame_persistence, args.mmf_metadata,
                 args.shared_transport, args.batched_shared_copy,
                 args.same_copy_native_authority, args.source_timing))
            or args.image_format != 'bmp' or args.ai_steer_multiplier != .9
            or args.flight_shadow_config
            or (args.duration_seconds, args.speed_hint_kmh, args.pace_task) not in
               ((3, 60, False), (30, 100, True))):
        raise ValueError('Explicit3s60/30s pace100 MMF diagnostic required')
    actor_profile = load_actor_profile(args.actor_cadence_profile)
    camera_status_profile = load_profile(args.camera_status_profile)
    if actor_trace_sink is None:
        raise ValueError('Explicit same-parent RAM actor trace sink required')
    actor_trace = ActorTrace()  # fixed capacity allocated before activation
    frame_writer_factory = HostFrameWriterFactory(profile)
    if bool(args.flight_shadow_config) != bool(args.flight_shadow_verification):
        raise ValueError('Shadow configuration and successful replay must be supplied together')
    if args.shared_transport and args.image_format != 'bmp':
        raise ValueError('Shared transport requires explicit lossless host BMP')
    if args.async_pixel_precompute and not args.shared_transport:
        raise ValueError('Async cache profile requires immutable shared transport')
    if args.batched_shared_copy and not args.shared_transport:
        raise ValueError('Batched copy requires immutable shared transport')
    if args.async_frame_persistence and (not args.batched_shared_copy or args.flight_shadow_config):
        raise ValueError('Async persistence requires batched transport; teacher calibration only')
    if args.mmf_metadata and (not args.async_frame_persistence or args.flight_shadow_config):
        raise ValueError('MMF v3 requires bounded async persistence and teacher calibration only')
    if args.source_timing and (not args.mmf_metadata or args.flight_shadow_config):
        raise ValueError('Source timing requires separate MMF teacher diagnostic without shadow')
    if (args.brake_pulse or args.pace_task) and args.flight_shadow_config:
        raise ValueError('Brake intervention is diagnostic only, not a qualified shadow task')
    if args.speed_hint_kmh == 60 and args.flight_shadow_config:
        raise ValueError('No high-speed flight shadow qualification yet')
    command = expert_command(args.duration_seconds, args.speed_hint_kmh, args.recover_to15,
                             args.lateral_recovery, args.brake_pulse, args.pace_task)
    if ((args.duration_seconds == 30 or args.speed_hint_kmh >= 30)
            and args.ai_steer_multiplier != .9):
        raise ValueError('Longer/faster expert profile requires the bounded AI steer .9 setting')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    if config.get('camera_mount_contract') is not None and args.flight_shadow_config:
        raise ValueError('Exterior-front layout needs independent shadow/model qualification')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862953,
                                     sensory_configuration=config,
        camera_status_profile=camera_status_profile,
        camera_status_profile_sha256=sha256(args.camera_status_profile),
        actor_cadence_profile=actor_profile.record(),
        actor_cadence_profile_sha256=sha256(args.actor_cadence_profile),
        frame_persistence_profile=profile.record(),
        frame_persistence_profile_source_sha256=sha256(args.frame_persistence_profile)))
    result = dict(status='failed', training_admissible=False, expert_labels_available=False,
                  teacher_kind='engine_ai_explicit_pace_task' if args.pace_task else (
                      'engine_ai_brake_intervention' if args.brake_pulse else 'engine_ai'),
                  actor_input_pose_free=True,
                  ui_actions_sent=0, vjoy_messages_sent=0, goal_achieved=False)
    result['frame_persistence_profile'] = profile.record()
    result['actor_cadence_profile'] = actor_profile.record()
    result['camera_status_profile'] = camera_status_profile
    vision, child, log, shadow = None, None, None, None
    shadow_results = []
    observations, times, camera_gaps = [], [], []
    authority_records, authority_eligible_records = 0, 0
    events_path = run / 'events.jsonl'
    if args.local_event_spool:
        spool = Path(os.environ['LOCALAPPDATA']) / 'flybrain/event-spool' / uuid.uuid4().hex
        spool.mkdir(parents=True, exist_ok=False)
        events_path = spool / 'events.jsonl'
    try:
        if args.same_copy_native_authority:
            proof = preserve_native_authority_sources(run)
            result['same_copy_native_authority'] = dict(enabled=True,
                schema='flybrain.native.same_copy_teacher_authority.v1',
                source_proof_sha256=sha256(run/'native-authority-source-proof.json'),
                executed_program_sha256=proof['executed_program_sha256'],
                fit_permission_granted=False)
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one reviewed engine required')
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / 'identity.json', identity)
        expected = dict(car_model='bmw_z4_gt3', track='monza',
                        shared_memory_version='1.7', ac_version='1.16.4')
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError('Reviewed native car/track/version identity required')
        if args.flight_shadow_config:
            from flybrain_drive.flight_stream import FlightStream

            shadow_config, shadow_proof = verified_shadow(args.flight_shadow_config,
                                                          args.flight_shadow_verification)
            if (args.recover_to15 or args.lateral_recovery
                    or shadow_config['task_speed_kmh'] != args.speed_hint_kmh):
                raise ValueError('Shadow task speed must match the fixed owned AI profile')
            import torch

            torch.set_num_threads(1)
            torch.manual_seed(shadow_proof['initialization_seed'])
            shadow = FlightStream(args.flight_shadow_config, native_identity=identity,
                                  episode_id=run.name)
            write_json(run/'flight-shadow-proof.json', dict(shadow_proof,
                checkpoint_sha256=shadow_config['checkpoint_sha256'],
                student_actuator_messages_sent=0, actor_qualified=False))
        retina = CSPCameraRetina(config)
        if heartbeat_start is not None:
            # Explicit caller-owned arm after begin_run/Git/shadow work, before
            # camera setup. Default callers create no heartbeat context/worker.
            result['heartbeat_transport_start'] = heartbeat_start()
        vision = CSPVisionRecorder(windows[0].hwnd, config,
                                   max_frames=1200 if args.duration_seconds == 30 else 350,
                                   max_bytes=1024**3,
                                   pixel_preprocessor=retina.precompute_pixels,
                                   image_format=args.image_format,
                                   shared_transport=args.shared_transport,
                                   batched_shared_copy=args.batched_shared_copy,
                                   async_frame_persistence=args.async_frame_persistence,
                                   frame_writer_factory=frame_writer_factory,
                                   host_copy_profile=profile,
                                   mmf_metadata=args.mmf_metadata,
                                   source_timing_v1=args.source_timing,
                                   camera_status_profile=camera_status_profile,
                                   async_pixel_precompute=args.async_pixel_precompute,
                                   period=.04 if args.shared_transport else .03)
        vision.wait_ready()
        log = (run/'control.log').open('x', encoding='utf8')
        child_arguments = [sys.executable, '-m', 'scripts.control_ac_session',
            '--command', command, '--execute',
            '--camera-request-id', vision.request_id,
            '--ai-steer-multiplier', str(args.ai_steer_multiplier), '--output', str(run/'control')]
        deadline = time.monotonic()+args.duration_seconds+(13 if args.pace_task else 10)
        session_status = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'

        def ai_is_running():
            return read_ai_phase(session_status, vision.clock.generation)

        with native_recording_reader(args.same_copy_native_authority,
                                     clock=vision.host_clock) as reader:
            # Prime this same reader before activation. Its first snapshot can be a
            # paused packet; it is a baseline, not a newly advanced actor observation.
            baseline, baseline_authority = receive_recording_native(
                reader, same_copy_native_authority=args.same_copy_native_authority)
            if (ai_is_running() or abs(baseline.fields['speedKmh']) >= .1
                    or baseline.fields['gas'] >= .02 or baseline.fields['brake'] <= .98):
                raise ValueError('Stopped preactivation native baseline required')
            write_json(run/'native-before-activation.json', physics_record(baseline))
            result['native_baseline_packet_id'] = baseline.packet_id
            if baseline_authority is not None:
                write_json(run/'native-before-activation-authority.json', native_authority_metadata(
                    baseline_authority, baseline_packet=baseline.packet_id,
                    decision_time=baseline.received_monotonic_seconds, csp_running=False,
                    maximum_body_age=retina.max_age, teacher_kind=result['teacher_kind']))
            from flybrain_drive.session_heartbeat_admission import require_leased_collection_budget

            # Last check after wait_ready, baseline and their metadata I/O. No
            # deadline refresh, extra ping or persistence between this and Popen.
            heartbeat_budget = require_leased_collection_budget(
                session_status, vision.clock.generation, args.duration_seconds)
            if heartbeat_budget is not None:
                result['heartbeat_activation_budget'] = heartbeat_budget
            child = subprocess.Popen(child_arguments, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            with events_path.open('x', encoding='utf8') as stream:
                while True:
                    actor_trace.begin_iteration()
                    try:
                        started = time.perf_counter()
                        actor_trace.budget_started(started)
                        with actor_trace.span('child_poll'):
                            child_active = child.poll() is None
                        if not child_active:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Same owned motion probe exceeded observation deadline')
                        with actor_trace.span('status'):
                            held_task = read_ai_phase(session_status, vision.clock.generation,
                                task=True, clock=vision.host_clock) if args.pace_task else None
                            running = held_task is not None if args.pace_task else ai_is_running()
                        if not running:
                            # Paused physics legitimately publishes no new packets. Wait for
                            # the owned AI phase; never re-date and pack a frozen snapshot.
                            with actor_trace.span('sleep'):
                                actor_trace.sleep_requested(.01)
                                time.sleep(.01)
                            continue
                        try:
                            with actor_trace.span('candidate_wait'):
                                vision.wait_for_candidate(timeout=.04)
                        except NoFreshCSPFrame as error:
                            camera_gaps.append(dict(host_clock=vision.host_clock(), reason=str(error)))
                            continue  # preserve the gap; never fabricate a fresh actor observation
                        try:
                            with actor_trace.span('native_receive'):
                                body, coupled_authority = receive_recording_native(
                                    reader, same_copy_native_authority=args.same_copy_native_authority)
                        except TimeoutError:
                            with actor_trace.span('status'):
                                still_running = ai_is_running()
                            if not still_running:
                                continue
                            raise
                        try:
                            with actor_trace.span('candidate_select'):
                                candidate = vision.candidate_before(body.received_monotonic_seconds)
                        except NoFreshCSPFrame as error:
                            camera_gaps.append(dict(host_clock=vision.host_clock(), reason=str(error)))
                            continue
                        decision = vision.host_clock()
                        actor_trace.decision(len(times), decision)
                        with actor_trace.span('retinal_pack'):
                            packed = retina.pack(vision.samples_for(candidate), decision_time=decision,
                                native_scalars={k: body.fields.get(k)
                                                for k in retina.required_native_fields},
                                native_receipt=body.received_monotonic_seconds)
                        observations.append(packed)
                        times.append(decision)
                        event = dict(index=len(times)-1, camera=candidate,
                            native=physics_record(body), decision_monotonic_seconds=decision,
                            foreground=bool(native.user.GetForegroundWindow() == windows[0].hwnd))
                        if coupled_authority is not None:
                            event['native_teacher_authority'] = native_authority_metadata(
                                coupled_authority, baseline_packet=baseline.packet_id,
                                decision_time=decision, csp_running=running,
                                maximum_body_age=retina.max_age, teacher_kind=result['teacher_kind'])
                        if args.pace_task:
                            event['external_task_at_receipt'] = held_task
                        shadow_error = None
                        if shadow is not None:
                            from flybrain_drive.virtual_body import decode_virtual_body_prefix

                            try:
                                pixels = {name: sample['rgb'] for name, sample in
                                          vision.samples_for(candidate).items()}
                                response = shadow.advance(clock=decision, camera=candidate,
                                    pixels=pixels, native_sample=decode_virtual_body_prefix(
                                        body.prefix_bytes, body.received_monotonic_seconds))
                                response['source_age_upper_at_output_seconds'] = max(
                                    response['output_host_clock']-v['render_started_lower']
                                    for v in candidate['views'])
                                response['body_values'] = shadow.last_body.values.tolist()
                                response['body_valid'] = shadow.last_body.valid.tolist()
                                event['flight_shadow'] = response
                                shadow_results.append(response)
                            except Exception as error:
                                shadow_error = dict(error=type(error).__name__, reason=str(error))
                                event['flight_shadow_error'] = shadow_error
                        # Preserve the complete offending observation on shadow failure as well.
                        with actor_trace.span('event_encode'):
                            event_json = json.dumps(event, allow_nan=False)+'\n'
                        with actor_trace.span('event_write'):
                            stream.write(event_json)
                        with actor_trace.span('event_flush'):
                            stream.flush()
                        if coupled_authority is not None:
                            authority_records += 1
                            authority_eligible_records += int(
                                event['native_teacher_authority']['teacher_authority_eligible'])
                        if shadow_error is not None:
                            raise RuntimeError('Coupled shadow inference failed: '+str(shadow_error))
                        with actor_trace.span('sleep'):
                            sleep_seconds = max(0., actor_profile.nominal_seconds-(time.perf_counter()-started))
                            actor_trace.sleep_requested(sleep_seconds)
                            time.sleep(sleep_seconds)
                    finally:
                        actor_trace.finish_iteration(
                            original_error_active=sys.exc_info()[0] is not None)
        if child.returncode != 0:
            raise RuntimeError('Owned motion probe failed; inspect control receipts')
        if not observations:
            raise ValueError('No fresh four-view/body observations during owned AI phase')
        receipt = json.loads((run/'control/api-receipt.json').read_text(encoding='utf8'))
        result.update(status='completed', camera_request_id=vision.request_id,
                      maximum_speed_kmh=receipt['maximum_speed_kmh'],
                      maximum_position_delta_m=receipt['maximum_position_delta_m'])
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        if vision and vision.triggered:
            (vision.directory/'stop').touch(exist_ok=True)
            result['owned_camera_stop_requested'] = True
        if child and child.poll() is None:
            request = run/'control/request.json'
            if request.exists():
                owned = json.loads(request.read_text(encoding='utf8'))
                stop = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'/(owned['id']+'.stop')
                stop.touch(exist_ok=False)
                result['owned_cancel_requested'] = True
            # Observe the same finite server lease; do not restart or issue a second activation.
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                result['control_process_still_live'] = child.pid
    finally:
        if log:
            log.close()
        if observations:
            np.savez_compressed(run/'observations.npz',
                **{'obs_'+k: np.stack([o[k] for o in observations]) for k in observations[0]},
                observation_time=np.array(times))
        result['decisions'] = len(times)
        if args.same_copy_native_authority:
            result.setdefault('same_copy_native_authority', dict(enabled=True)).update(
                records=authority_records, eligible_records=authority_eligible_records,
                ineligible_records=authority_records-authority_eligible_records,
                actor_fields_changed=False, camera_teacher_authority_synchronized=False,
                fit_permission_granted=False)
        if shadow is not None:
            result['flight_shadow'] = dict(inferences=len(shadow_results),
                all_observations_inferred=bool(times) and len(shadow_results) == len(times),
                processing_seconds_percentiles=np.percentile(
                    [r['processing_seconds'] for r in shadow_results], [50, 95, 99, 100]).tolist()
                    if shadow_results else [],
                output_source_age_seconds_percentiles=np.percentile(
                    [r['source_age_upper_at_output_seconds'] for r in shadow_results],
                    [50, 95, 99, 100]).tolist() if shadow_results else [],
                visual_ticks=shadow.visual_ticks, flight_ticks=shadow.flight_ticks,
                failed_stream=shadow.failed, hidden_resets=0, student_actuator_messages_sent=0,
                online_sensory_inference_observed=bool(shadow_results), actor_qualified=False)
        result['camera_wait_misses'] = len(camera_gaps)
        write_json(run/'camera-gaps.json', camera_gaps)
        result['all_four_views_valid_decisions'] = sum(
            bool(o['view_valid'].all()) for o in observations)
        if vision:
            result['vision'] = vision.close()
            result['vision_archive'] = vision.archive_to(run/'vision', bundle=True)
            if vision.error and result['status'] == 'completed':
                result.update(status='failed', reason='Camera worker failed: '+str(vision.error))
        if args.local_event_spool and events_path.exists():
            expected_hash = sha256(events_path)
            shutil.copyfile(events_path, run / 'events.jsonl')
            if sha256(run / 'events.jsonl') != expected_hash:
                raise ValueError('Original local event copy hash mismatch')
            write_json(run / 'local-event-spool.json', dict(
                source=str(events_path), destination=str(run / 'events.jsonl'),
                sha256=expected_hash, source_retained=True, bytes=events_path.stat().st_size))
        result['owned_cleanup'] = recorder_cleanup(child, vision)
        actor_trace_sink.accept(actor_trace.seal(), result['owned_cleanup'])
        result['actor_trace_decisions'] = actor_trace.decisions
        write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
