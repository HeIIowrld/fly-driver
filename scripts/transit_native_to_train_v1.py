"""Finite native-only diagnostic transit; inventory/default, ROOT preflight and execute separate."""

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from flybrain_drive.native_transit_observer_v1 import finite, observe_paused, sha

SCHEMA = 'flybrain.finite_native_only_train_start_transit.v1'
CHILD_MODULE = 'scripts.control_ac_session'
LUA = 'integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'
ABI = 'assets/external/ac-native-memory-20261027/upstream/sim_info.py'
ABI_METADATA = 'assets/external/ac-native-memory-20261027/source.json'
RUNTIME = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
PRIOR_ROOT = RUNTIME/'20261030-t1837-native-only-motion60-qualification'
PRIOR_RESULT_SHA256 = 'f8fc37f803eaf7f0fa8d20a1d63c96d32e6b10a7462f881d366f47ec6789df57'
PRIOR_REVIEW = RUNTIME/'20261030-t1838-native-only-motion60-review/actual-supplement/result.json'
PRIOR_REVIEW_SHA256 = '61ec87d0151a1263b044607609e1138c2da21cfa47c90ce43036ae09468fd9a5'
ROOT_TERMINAL_BASIS = dict(origin='parent original authoritative tool result',
    session_id=2103, chunk_id='6899af', terminal_exit_code=0)
PRIOR_FILES = ('result.json', 'manifest.json', 'executed-program.py', 'control/result.json',
    'control/request.json', 'control/api-receipt.json', 'control/native-after.json',
    'control/after.json', 'control/native-motion-samples.json',
    'after-native-observer/static200.bin',
    'after-native-observer/raw-read-0.json', 'after-native-observer/raw-read-1.json',
    'after-native-observer/sample-0.json', 'after-native-observer/sample-1.json',
    'after-native-observer/graphics-0-264.bin', 'after-native-observer/graphics-1-264.bin',
    'after-native-observer/physics-AI-0-420.bin', 'after-native-observer/physics-AI-1-420.bin')
NATIVE_IMPORT_SOURCES = (
    'scripts/__init__.py', 'scripts/control_ac_session.py', 'scripts/record_ac_wgc.py',
    'src/flybrain_drive/__init__.py', 'src/flybrain_drive/ac_capture.py',
    'src/flybrain_drive/ac_image_buffer.py', 'src/flybrain_drive/ac_lifecycle_commit.py',
    'src/flybrain_drive/ac_native_clock.py', 'src/flybrain_drive/ac_native_physics.py',
    'src/flybrain_drive/ac_native_position.py', 'src/flybrain_drive/ac_published_state.py',
    'src/flybrain_drive/ac_teacher_physics.py', 'src/flybrain_drive/ac_teacher_status.py',
    'src/flybrain_drive/ac_vjoy.py', 'src/flybrain_drive/ac_wgc.py',
    'src/flybrain_drive/ac_windows_session.py', 'src/flybrain_drive/episode_storage.py',
    'src/flybrain_drive/evidence.py', 'src/flybrain_drive/graph.py',
    'src/flybrain_drive/session_heartbeat_admission.py',
    'src/flybrain_drive/session_heartbeat_candidate.py',
    'src/flybrain_drive/session_heartbeat_provenance.py', 'src/flybrain_drive/teacher_pace.py',
)
MANDATORY = (*NATIVE_IMPORT_SOURCES, LUA, ABI, ABI_METADATA, 'configs/ac-axis-probe.json',
    'src/flybrain_drive/native_transit_observer_v1.py', 'scripts/transit_native_to_train_v1.py',
    'scripts/prepare_native_transit_observer_v1.py', 'scripts/transit_to_train_start_v2.py')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def validate_plan(plan):
    prior = plan.get('prior_registered_activations')
    if (plan.get('schema') != SCHEMA or plan.get('role') != 'transit'
            or plan.get('fit_allowed') is not False or plan.get('automatic_retry') is not False
            or type(prior) is not int or prior != 1
            or plan.get('maximum_total_activations') != 30
            or plan.get('maximum_new_activations') != 30-prior
            or plan.get('command') != 'expert-motion-speed60' or plan.get('hold_seconds') != 3
            or plan.get('AI_steer_multiplier') != .9 or plan.get('hard_speed_kmh') != 80
            or plan.get('maximum_displacement_m') != 80 or plan.get('source_telemetry_cap') != 512
            or plan.get('published_CLI_observation_budget_seconds') != 14
            or plan.get('parent_process_wait_seconds') != 30
            or plan.get('target_start_region') != [.275, .44]
            or plan.get('engine_pid') != 24200 or plan.get('epoch') != 0
            or plan.get('generation') != '733364ULL:0.0014992'
            or plan.get('lua_sha256') != (
                'b6d87d2708a13b37e858bf6cfe87f362540bf645514223e18e2c870fbd66e1f7')
            or plan.get('native_graphics_pair_minimum_seconds') != .08
            or plan.get('observation_timeout_seconds') != 2.
            or plan.get('observer_source') != 'native_graphics.normalizedCarPosition@248'
            or plan.get('native_graphics_prefix_bytes') != 264
            or plan.get('minimum_free_disk_bytes') != 512*1024**2
            or not finite(plan.get('expected_initial_spline'))
            or not 0 <= plan['expected_initial_spline'] <= .44
            or set(MANDATORY)-set(plan.get('input_sha256', {}))):
        raise ValueError('Exact finite native-only diagnostic contract required')
    if any(not re.fullmatch('[0-9a-f]{64}', v) for v in plan['input_sha256'].values()):
        raise ValueError('Closed original source SHA256 bindings required')
    if not isinstance(plan.get('prior_qualification'), dict):
        raise ValueError('Original1837 qualification reference and remaining29 required')
    reference = plan['prior_qualification']
    if (Path(reference.get('result_file', '')).resolve() != (PRIOR_ROOT/'result.json').resolve()
            or reference.get('result_sha256') != PRIOR_RESULT_SHA256
            or Path(reference.get('control_directory', '')).resolve() != (
                PRIOR_ROOT/'control').resolve()
            or any(str(PRIOR_ROOT/p).replace('\\', '/') not in plan['input_sha256']
                   for p in PRIOR_FILES)):
        raise ValueError('All original1837 request/native-after/after-observer files must be bound')
    if (plan.get('prior_independent_review_file') != PRIOR_REVIEW.as_posix()
            or plan.get('prior_independent_review_sha256') != PRIOR_REVIEW_SHA256
            or plan.get('root_original_terminal_basis') != ROOT_TERMINAL_BASIS
            or plan['input_sha256'].get(PRIOR_REVIEW.as_posix()) != PRIOR_REVIEW_SHA256):
        raise ValueError('Actual1838supplement/root original terminal2103 basis must be bound')
    outputs = [Path(plan[k]) for k in ('output', 'preflight_output')]
    if (any(p.drive.upper() != 'C:' or not p.resolve().is_relative_to(RUNTIME.resolve())
            for p in outputs) or outputs[0].resolve() == outputs[1].resolve()):
        raise ValueError('Separate new local-C owned output/preflight directories required')


def verify_inputs(plan, plan_file, plan_sha):
    if sha(plan_file) != plan_sha:
        raise ValueError('Sealed plan changed; no repin')
    for file, expected in plan['input_sha256'].items():
        if sha(file) != expected:
            raise ValueError('Declared native/control source changed: '+file)
    axis = json.loads(Path('configs/ac-axis-probe.json').read_bytes())
    if plan['input_sha256'].get(axis['dll_path']) != axis['dll_sha256']:
        raise ValueError('Actual SDK DLL must be explicitly pinned')
    if plan['input_sha256'][LUA] != plan['lua_sha256']:
        raise ValueError('Repository/installed source binding differs')
    if shutil.disk_usage('C:/').free < plan['minimum_free_disk_bytes']:
        raise ValueError('Finite native diagnostic storage reserve exhausted')


def decision(point):
    if not finite(point) or not 0 <= point < 1:
        raise ValueError('Original native normalized route required')
    if .275 <= point <= .44:
        return 'stop_reached'
    if point > .44:
        raise ValueError('Overshoot; no next lap or enlarged target')
    return 'continue'


def child_arguments(directory):
    return [sys.executable, '-m', CHILD_MODULE, '--command', 'expert-motion-speed60',
            '--ai-steer-multiplier', '.9', '--output', str(directory), '--execute']


def wait_native_child(process, result, *, seconds):
    """An unknown original child is retained; never terminate/kill or start another."""
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired as error:
        result['unknown_child_outcome'] = True
        result['timeout_original_child_pid'] = process.pid
        result['owned_child_still_live'] = process.pid
        raise RuntimeError('Original child outcome unknown; never kill/retry/continue') from error


def verify_observer_state(state):
    from scripts.control_ac_session import validate_expert_probe

    validate_expert_probe(state['samples'][-1]['status'], motion=True, collection=False,
                          fast=True, speed60=True)
    if not all(state.get(k) is True for k in (
            'FREE_background', 'same_frozen_physics', 'graphics_advance_observed')):
        raise ValueError('Complete original readonly native paused observer required')


def verify_child(directory, plan):
    """Read originals only; native HOST copies including frozen packets remain intact."""
    from flybrain_drive.ac_native_physics import decode_physics_prefix
    from scripts.control_ac_session import validate_probe_receipt

    directory = Path(directory)
    result = json.loads((directory/'result.json').read_bytes())
    request = json.loads((directory/'request.json').read_bytes())
    receipt = json.loads((directory/'api-receipt.json').read_bytes())
    after = json.loads((directory/'after.json').read_bytes())
    if (result.get('status') != 'completed' or result.get('lifecycle_requests_sent') != 1
            or result.get('lifecycle_publication_attempts') != 1
            or result.get('actuator_messages_sent') != 0 or result.get('ui_actions_sent') != 0
            or result.get('foreground_before') is not False
            or result.get('foreground_after') is not False
            or request.get('command') != plan['command'] or request.get('hold_seconds') != 3
            or request.get('generation') != plan['generation']
            or request.get('epoch') != plan['epoch']
            or request.get('speed_limit_kmh') != 60 or request.get('ai_steer_multiplier') != .9
            or 'camera_request_id' in request
            or receipt.get('id') != request.get('id') or receipt.get('command') != plan['command']
            or receipt.get('status') != 'api_returned_true'
            or receipt.get('motion_hard_speed_kmh') != 80
            or receipt.get('maximum_displacement_guard_m') != 80
            or not 2 <= len(receipt.get('telemetry', [])) <= 512):
        raise ValueError('Original single native-only3s60 publication/caps/outcome gate failed')
    validate_probe_receipt(receipt, motion=True, collection=False, fast=True, speed60=True)
    snapshots = [receipt['before'], *receipt['telemetry'], receipt['after']]
    for value in (receipt['before'], receipt['after'], after):
        if (value.get('generation') != plan['generation'] or value.get('epoch') != plan['epoch']
                or value.get('camera_active') is not False):
            raise ValueError('Original same-plant camera-absent receipt required')
    if (after.get('paused') is not True or after.get('ai_controlled') is not False
            or after.get('expert_probe_active') is not False
            or not finite(after.get('script_clock'))
            or any(s.get('wheels_outside') != 0 for s in snapshots)):
        raise ValueError('Completed paused AIoff/on-track task required before continuation')
    final = json.loads((directory/'native-after.json').read_bytes())
    raw = base64.b64decode(final['prefix_base64'], validate=True)
    sample = decode_physics_prefix(raw, final['received_monotonic_seconds'])
    if (sample.packet_id != final['packet_id']
            or hashlib.sha256(raw).hexdigest() != final['prefix_sha256']
            or abs(sample.fields['speedKmh']) >= .1 or not 0 <= sample.fields['gas'] < .02
            or not .98 < sample.fields['brake'] <= 1):
        raise ValueError('Original native final fullbrake required')
    native_rows = json.loads((directory/'native-motion-samples.json').read_bytes())
    if not isinstance(native_rows, list) or not native_rows:
        raise ValueError('All original native HOST copies required')
    packets, times = [], []
    for row in native_rows:
        data = base64.b64decode(row['prefix_base64'], validate=True)
        decoded = decode_physics_prefix(data, row['received_monotonic_seconds'])
        if (hashlib.sha256(data).hexdigest() != row['prefix_sha256']
                or decoded.packet_id != row['packet_id']):
            raise ValueError('Original native HOST sample bytes/receipt differ')
        packets.append(decoded.packet_id)
        times.append(decoded.received_monotonic_seconds)
    if any(a > b for a, b in zip(packets, packets[1:], strict=False)) or any(
            a > b for a, b in zip(times, times[1:], strict=False)):
        raise ValueError('Original native HOST copy sequence reversed')
    return dict(after_source=after['script_clock'],
        before_route=receipt['before']['diagnostic_route'],
        after_route=receipt['after']['diagnostic_route'],
        native_after_prefix_sha256=final['prefix_sha256'],
        native_after_packet=sample.packet_id, result_sha256=sha(directory/'result.json'),
        request_sha256=sha(directory/'request.json'),
        receipt_sha256=sha(directory/'api-receipt.json'),
        native_motion_sha256=sha(directory/'native-motion-samples.json'),
        native_HOST_copies=len(native_rows), native_HOST_unique_packets=len(set(packets)),
        source_telemetry_samples=len(receipt['telemetry']), original_samples_retained=True,
        new_actor_or_teacher_labels=False, fit_allowed=False)


def post_child(before, after, gate):
    old, current = before['observer'], after['observer']
    if current['spline_position'] < old['spline_position']-.001:
        raise ValueError('Wrap/reverse; stop without new activation')
    for route, observer in ((gate['before_route'], old), (gate['after_route'], current)):
        if (route.get('actor_input') is not False or route.get('lap_count') != observer['lap_count']
                or abs(route['spline_position']-observer['spline_position']) > 1e-4):
            raise ValueError('Actual task/native paused route applicability differs')
    last = after['samples'][-1]
    raw = Path(last['physics420_file']).read_bytes()
    if (hashlib.sha256(raw[:308]).hexdigest() != gate['native_after_prefix_sha256']
            or last['native_packet'] != gate['native_after_packet']):
        raise ValueError('Final CLI/native observer frozen-body attachment differs')
    decision(current['spline_position'])


def execute_stages(plan, result, *, observe, child, verify, checkpoint, prior_gate=None):
    state = observe(None, 0)
    if abs(state['observer']['spline_position']-plan['expected_initial_spline']) > 1e-4:
        raise ValueError('Fresh original start differs from plan; no activation')
    first_qualified = prior_gate is not None
    result['camera_absent_branch_actual_qualified'] = first_qualified
    for index in range(plan['maximum_new_activations']):
        if decision(state['observer']['spline_position']) == 'stop_reached':
            return dict(status='completed_native_transit_reached_registered_start',
                        final_spline=state['observer']['spline_position'],
                        staging_start_reached=True)
        verify()
        row = dict(index=index, overall_activation_index=plan['prior_registered_activations']+index,
            role='transit', fit_allowed=False, status='before_activation', before_observer=state)
        result['attempts'].append(row)
        result['overall_activation_slots_reserved'] = plan['prior_registered_activations']+index+1
        checkpoint()
        gate = child(index)
        row.update(status='child_terminal_gates_passed', original_child_gate=gate)
        checkpoint()
        current = observe(gate['after_source'], index+1)
        post_child(state, current, gate)
        row.update(after_observer=current, status='completed_original_native_only_diagnostic')
        first_qualified = True
        result['camera_absent_branch_actual_qualified'] = True
        result['overall_activations_consumed'] = plan['prior_registered_activations']+index+1
        checkpoint()
        state = current
    point = state['observer']['spline_position']
    return dict(status='completed_native_transit_reached_registered_start' if (
        decision(point) == 'stop_reached') else 'completed_finite_native_transit_start_not_reached',
        final_spline=point, staging_start_reached=decision(point) == 'stop_reached',
        camera_absent_branch_actual_qualified=first_qualified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight-only', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-result-sha256')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    validate_plan(plan)
    plan_sha = sha(args.plan)
    verify_inputs(plan, args.plan, plan_sha)
    if not args.preflight_only and not args.execute:
        print(json.dumps(dict(status='inventory_no_live_interfaces', plan_sha256=plan_sha,
            remaining_activations=plan['maximum_new_activations'],
            role='transit', fit_allowed=False)))
        return
    target = Path(plan['preflight_output'] if args.preflight_only else plan['output'])
    target.mkdir(parents=True, exist_ok=False)
    write(target/'registration-before-live.json', dict(plan=plan, plan_sha256=plan_sha,
        executed_program_sha256=sha(__file__), role='transit', fit_allowed=False))
    (target/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', role='transit', fit_allowed=False, optimizer_updates=0,
        plan_sha256=plan_sha, attempts=[],
        overall_activations_consumed=plan['prior_registered_activations'],
        new_camera_requests=0, new_heartbeat_leases=0, new_GPU_or_model_work=0,
        goal_achieved=False,
        automatic_retries=0, child_termination_calls=0, owned_child_still_live=None)
    live_child = None
    try:
        if args.preflight_only:
            observed = observe_paused(plan, target/'observer')
            verify_observer_state(observed)
            verify_inputs(plan, args.plan, plan_sha)
            result.update(status='completed_native_only_readonly_preflight', observer=observed)
            return
        preflight = Path(plan['preflight_output'])/'result.json'
        if not args.preflight_result_sha256 or sha(preflight) != args.preflight_result_sha256:
            raise ValueError('Explicit original successful ROOT readonly-preflight SHA required')
        original = json.loads(preflight.read_bytes())
        if (original['status'] != 'completed_native_only_readonly_preflight'
                or original['plan_sha256'] != plan_sha or original.get('role') != 'transit'
                or original.get('fit_allowed') is not False or original.get('attempts') != []
                or sha(preflight.parent/'executed-program.py') != plan['input_sha256'][
                    'scripts/transit_native_to_train_v1.py']):
            raise ValueError('Preflight belongs to another plan/outcome')
        prior_gate = None
        if plan['prior_registered_activations']:
            prior = plan['prior_qualification']
            if sha(prior['result_file']) != prior['result_sha256']:
                raise ValueError('Original1837 result binding differs')
            parent = json.loads(Path(prior['result_file']).read_bytes())
            if (parent.get('status') != 'completed_single_native_only_motion60_diagnostic'
                    or parent.get('fit_allowed') is not False or parent.get('role') != 'transit'
                    or parent.get('teacher_activations') != 1
                    or any(parent.get(k) != 0 for k in ('camera_requests_created',
                        'heartbeat_contexts_created', 'GPU_actions', 'UI_actions',
                        'driver_acquisitions'))):
                raise ValueError('Original1837 qualification must complete before continuation')
            review = json.loads(Path(plan['prior_independent_review_file']).read_bytes())
            if (sha(plan['prior_independent_review_file']) != PRIOR_REVIEW_SHA256
                    or review.get('status') != 'passed_saved_native_API_diagnostic_evidence'
                    or review.get('failed_checks') != 0 or review.get('motion_activations') != 1
                    or review.get('source_result_sha256') != PRIOR_RESULT_SHA256
                    or review.get('actual_tool_basis') != ROOT_TERMINAL_BASIS):
                raise ValueError('Original actual independent qualification/terminal basis differs')
            prior_gate = verify_child(prior['control_directory'], plan)
            if prior_gate['result_sha256'] != prior['control_result_sha256']:
                raise ValueError('Original1837 child does not qualify remaining29')
            post_child(parent['before_observer'], parent['after_observer'], prior_gate)
            if abs(parent['after_observer']['observer']['spline_position']-plan[
                    'expected_initial_spline']) > 1e-7:
                raise ValueError('Plan must use fresh original1837 AFTER start, never old.096418')

        def checkpoint():
            write(target/'result.json', result)

        def observe(after_source, index):
            minimum = (original['observer']['samples'][-1]['status']['script_clock']
                       if index == 0 else after_source)
            value = observe_paused(plan, target/f'observer-{index:02d}', after_source=minimum)
            verify_observer_state(value)
            if index == 0:
                previous = original['observer']['samples'][-1]
                now = value['samples'][-1]
                if now['physics420_sha256'] != previous['physics420_sha256']:
                    raise ValueError('Paused baseline changed after ROOT preflight')
            return value

        def child(index):
            nonlocal live_child
            directory = target/f'attempt-{index:02d}'
            directory.mkdir()
            write(directory/'role-before-activation.json', dict(role='transit', fit_allowed=False,
                plan_sha256=plan_sha,
                overall_activation_index=plan['prior_registered_activations']+index))
            control = directory/'control'
            with (directory/'child.log').open('x', encoding='utf8') as log:
                live_child = subprocess.Popen(child_arguments(control), stdout=log,
                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                result['attempts'][-1]['original_child_pid'] = live_child.pid
                checkpoint()
                code = wait_native_child(live_child, result,
                                         seconds=plan['parent_process_wait_seconds'])
            result['attempts'][-1]['terminal_exit_code'] = code
            checkpoint()
            if code != 0:
                raise RuntimeError('Original native-only child failed; no next activation')
            return verify_child(control, plan)

        outcome = execute_stages(plan, result, observe=observe, child=child,
            verify=lambda: verify_inputs(plan, args.plan, plan_sha), checkpoint=checkpoint,
            prior_gate=prior_gate)
        verify_inputs(plan, args.plan, plan_sha)
        result.update(outcome)
    except BaseException as error:
        result.update(status='failed', error=type(error).__name__, reason=str(error))
        raise
    finally:
        if live_child is not None:
            terminal = live_child.poll()
            result['child_poll_at_parent_finally'] = terminal
            result['owned_child_still_live'] = live_child.pid if terminal is None else None
        write(target/'result.json', result)


if __name__ == '__main__':
    main()
