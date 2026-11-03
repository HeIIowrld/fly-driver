"""Explicit finite live qualifier; default prints inventory and performs no live I/O.

Root reviews/deploys the separate candidate first. No deployment, reset, student,
GPU, extra vehicle command, stale-owner takeover or lifetime refresh is included.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.pedal_train_candidate import collector_declaration

CANDIDATE = Path('integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua')


def frozen_native_baseline(snapshot):
    """Paused last-packet flag is diagnostic, never fresh student authority."""
    if type(snapshot.get('builtin_ai_controlled')) is not bool:
        raise ValueError('Explicit decoded native flag required as frozen baseline')
    return dict(packet_id=snapshot['packet_id'],
        builtin_ai_controlled=snapshot['builtin_ai_controlled'],
        prefix_sha256=snapshot['prefix_sha256'],
        received_monotonic_seconds=snapshot['received_monotonic_seconds'],
        frozen_paused_packet=True, fresh_AI_ownership_claimed=False, actor_input=False)


def collection_arguments(mode, output, profile_file):
    if mode not in ('camera3', 'pace30'):
        raise ValueError('Only reviewed3second60 or30second pace diagnostic')
    arguments = ['--config', 'configs/ac-csp-exterior-front-v3-retina.json',
        '--frame-persistence-profile', str(profile_file),
        '--image-format', 'bmp', '--shared-transport', '--batched-shared-copy',
        '--async-pixel-precompute', '--async-frame-persistence', '--mmf-metadata',
        '--source-timing', '--same-copy-native-authority', '--ai-steer-multiplier', '.9',
        '--duration-seconds', '30' if mode == 'pace30' else '3',
        '--speed-hint-kmh', '100' if mode == 'pace30' else '60', '--output', str(output)]
    if mode == 'pace30':
        arguments.append('--pace-task')
    return arguments


def wait_owned(context, predicate, seconds, description):
    deadline = min(time.perf_counter_ns()+int(seconds*1e9),
                   context.lease.registration['deadline_qpc_ns'])
    while time.perf_counter_ns() <= deadline:
        if context.error:
            raise RuntimeError(context.error)
        value = predicate()
        if value:
            return value
        time.sleep(.005)
    raise TimeoutError(description+'; never restart, extend or create another owner')


def run(args):
    sealed_train = collector_declaration(args)
    from flybrain_drive.ac_capture import WindowsACWindow
    from flybrain_drive.ac_native_physics import NativePhysicsReader
    from flybrain_drive.ac_published_state import read_published_json
    from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
    from flybrain_drive.ac_vjoy import load_vjoy_sdk
    from flybrain_drive.frame_persistence_profile_v3 import load_buffered_frame_profile
    from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
    from flybrain_drive.session_heartbeat_evidence_v1 import (
        EvidenceDeferred,
        HeartbeatEvidence,
        require_local_c_output,
    )
    from flybrain_drive.session_heartbeat_runtime import (
        HeartbeatSessionContext,
        initialize_qpc_calibration,
    )

    profile = load_buffered_frame_profile(args.frame_persistence_profile)
    require_local_c_output(args.output)
    evidence = HeartbeatEvidence()
    session_root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    camera_root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera'
    status_path = session_root/'status.json'
    refuse_leased_transport('new independent service qualification', status_path=status_path)
    if sha256(CANDIDATE) != args.expected_installed_sha256:
        raise ValueError('Reviewed separate candidate SHA differs')
    # NAS/Git/source manifest work precedes every possible60second lease start.
    output = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        role='train', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE),
        train_nonce=sealed_train['nonce'],
        train_declaration_sha256=sealed_train['sha256'],
        original_train_declaration=sealed_train['value']))
    with (output/'original-train-declaration.json').open('xb') as declaration_file:
        declaration_file.write(sealed_train['raw'])
    with (output/'executed-program.py').open('xb') as executed_source:
        executed_source.write(Path(__file__).read_bytes())
    result = dict(status='failed', role='train', fit_allowed=False, optimizer_updates=0,
        train_nonce=sealed_train['nonce'],
        train_declaration_sha256=sealed_train['sha256'],
        student_deployed=False, goal_achieved=False, deployed_by_this_program=False)
    result['frame_persistence_profile'] = profile.record()
    context = None

    def fresh_idle():
        value, modified = read_published_json(status_path)
        if (not 0 <= time.time()-modified <= 2
                or value.get('generation') != args.expected_generation
                or value.get('epoch') != args.expected_epoch
                or value.get('paused') is not True or value.get('ai_controlled') is not False
                or value.get('is_online') is not False
                or value.get('car_id') != 'bmw_z4_gt3' or value.get('track_id') != 'monza'
                or any(value.get(k) for k in ('camera_active', 'expert_probe_active',
                    'brake_settle_active', 'pending_receipt_id', 'error'))
                or abs(value.get('speed_kmh', 1e9)) >= .1
                or value.get('gas', 1.) >= .02 or value.get('brake', 0.) <= .98):
            raise ValueError('Fresh exact-generation/epoch paused stationary AIoff idle required')
        return value

    def start_transport():
        nonlocal context
        if context is not None:
            raise RuntimeError('Never retry heartbeat start in the same qualification')
        fresh_idle()
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1 or windows[0].pid != args.expected_engine_pid:
            raise ValueError('Exact original reviewed engine PID required')
        window = windows[0]
        installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        if sha256(installed) != args.expected_installed_sha256:
            raise ValueError('Installed candidate SHA differs; qualifier never deploys')
        if native.user.GetForegroundWindow() == window.hwnd:
            raise ValueError('Background engine required; no UI actions')
        axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        if sdk.GetVJDStatus(axis['device_id']) != 1:
            raise ValueError('FREE controller required; never acquire/steal')
        with NativePhysicsReader(timeout=.15, clock=time.perf_counter) as reader:
            sample = reader.receive()
        if (abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
                or sample.fields['brake'] <= .98):
            raise ValueError('Original native stationary full brake required')
        teacher = NativeTeacherStatusReader()
        try:
            native_baseline = frozen_native_baseline(teacher.snapshot())
        finally:
            teacher.close()
        calibration = initialize_qpc_calibration(camera_root, args.expected_generation, window.hwnd)
        fresh_idle()
        captured_calibration = evidence.capture_calibration(calibration)
        context = HeartbeatSessionContext(status_path, calibration,
            map_factory=evidence.map_factory, control_factory=evidence.control_factory)
        if captured_calibration != context.calibration.calibration_sha256:
            raise ValueError('Captured original calibration fingerprint differs')
        context.prepare()
        context.arm_plan(epoch=args.expected_epoch)
        plan = context.start()  # host_ready only after actual initial control publication
        plan['clock_rounding_margin_ns'] = context.calibration.rounding_margin_ns
        write_json(session_root/'heartbeat-transport.json', plan)
        evidence.capture_arm(session_root/'heartbeat-transport.json')
        wait_owned(context, lambda: context.last_ack and context.last_ack[1]['phase'] in
            ('armed', 'publishing') and context.writer.last_committed_sequence > 0,
            2., 'No original MMF snapshot committed')
        fresh_idle()  # this path now necessarily validates hash-bound source QPC proof
        return dict(owner_id=context.owner_id, generation=args.expected_generation,
            deadline_qpc_ns=context.lease.registration['deadline_qpc_ns'],
            calibration_sha256=context.calibration.calibration_sha256,
            native_flag_frozen_baseline=native_baseline)

    try:
        if args.mode == 'stationary-smoke':
            result['transport_start'] = start_transport()
        else:
            from scripts.record_ac_engine_ai_probe_buffered import main as record_main

            original_arguments = sys.argv
            try:
                sys.argv = ['record_ac_engine_ai_probe', *collection_arguments(
                    args.mode, output/'recording', args.frame_persistence_profile)]
                record_main(heartbeat_start=start_transport)
            finally:
                sys.argv = original_arguments
        def idle_after_cleanup():
            try:
                return fresh_idle()
            except ValueError:
                return None  # same bounded metadata wait; never a new control request

        wait_owned(context, idle_after_cleanup, 1., 'No verified idle after camera cleanup')
        context.request_retire()
        wait_owned(context, lambda: context.last_ack and context.last_ack[1]['phase'] == 'final'
            and context.writer.last_committed_sequence == context.last_ack[1].get('final_sequence'),
            2., 'Final quiescent original sequence not committed')
        context.fence_writer(join_timeout=2.)

        def sync_return():
            from flybrain_drive.session_heartbeat_terminal_v2 import observe_sync_return

            return observe_sync_return(context)

        wait_owned(context, sync_return, 2., 'No exact synchronous return acknowledgement')
        evidence.capture_sync(context)
        result['handoff'] = context.complete_sync_return()
        evidence.bind_handoff(result['handoff'])
        fresh_idle()
        result['status'] = 'completed_stationary_transport_smoke' if (
            args.mode == 'stationary-smoke') else 'completed_train_candidate_transport_collection'
    except BaseException as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        if context is not None:
            if context.state != 'closed':
                context.stop_event.set()
                # No lease removal/sync fallback after uncertain I/O. Report the
                # same owned thread outcomes; a blocked writer can outlive this.
                for thread in (context.poll_thread, context.write_thread):
                    if thread is not None:
                        thread.join(timeout=.25)
            threads = (context.poll_thread, context.write_thread)
            rename_attempts = context.writer.rename_attempts if context.writer else {}
            result.update(transport_state=context.state, transport_error=context.error,
                worker_records=context.writer.records if context.writer else [],
                same_source_rename_attempts=rename_attempts,
                terminal_control_ack=getattr(context, 'terminal_control_ack', None),
                owned_lease_release=getattr(context, 'release_receipt', None),
                owned_threads_still_live=[t.name for t in threads
                    if t is not None and t.is_alive()],
                lease_retained=bool(context.lease and context.lease.path.exists()))
        try:
            result['heartbeat_evidence'] = evidence.persist(
                output/'heartbeat-evidence', context=context)
        except EvidenceDeferred as error:
            result['heartbeat_evidence'] = evidence.defer_persistence(
                output/'heartbeat-evidence', context=context,
                receipt_path=output/'heartbeat-evidence-terminal-receipt.json')
            result['heartbeat_evidence']['reason'] = str(error)
        except BaseException as error:
            result['heartbeat_evidence'] = dict(status='persistence_failed',
                error=type(error).__name__, reason=str(error),
                partial_owned_refs=evidence.pending_report(context))
        if (result['status'].startswith('completed')
                and result['heartbeat_evidence']['status'] != 'complete'):
            result.update(status='failed', error='IncompleteHeartbeatEvidence',
                reason='Original evidence incomplete; retain this run, never rerun automatically')
        write_json(output/'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-declaration', required=True)
    parser.add_argument('--mode', choices=['camera3', 'pace30'], required=True)
    parser.add_argument('--expected-engine-pid', type=int, required=True)
    parser.add_argument('--expected-generation', required=True)
    parser.add_argument('--expected-epoch', type=int, required=True)
    parser.add_argument('--expected-installed-sha256', required=True)
    parser.add_argument('--frame-persistence-profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(arguments=vars(args), live_actions=0, mode='inventory_only')))
        return
    run(args)


if __name__ == '__main__':
    main()
