"""Finite diagnostic transit with a separate GET-only native-observer preflight.

Default is inventory. The root-owned --preflight-only and --execute are distinct;
this module never deploys, resets, acquires a driver, or repairs a failed owner.
"""

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import struct
import sys
import time
from pathlib import Path

SCHEMA = 'flybrain.finite_diagnostic_native_geographic_transit.v2'
BASE_CHILD = 'scripts.qualify_session_heartbeat_evidence_buffered'
BASE_SHA = '50636a6bfd2577624683f92635905c91514e1b5e2261933842e2c36ee550dbb4'
ABI_SOURCE = 'assets/external/ac-native-memory-20261027/upstream/sim_info.py'
EXTRA_SOURCES = ('scripts/transit_to_train_start_v2.py',
                 'src/flybrain_drive/ac_native_position.py', ABI_SOURCE)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_plan(plan):
    from flybrain_drive.pedal_train_candidate import SOURCE_FILES

    if (plan.get('schema') != SCHEMA or plan.get('role') != 'transit'
            or plan.get('fit_allowed') is not False or plan.get('maximum_activations') != 3
            or plan.get('automatic_retry') is not False
            or plan.get('target_start_region') != [.275, .44]
            or plan.get('minimum_free_disk_bytes') != 8*1024**3
            or plan.get('epoch') != 0 or plan.get('engine_pid') != 24200
            or plan.get('generation') != '733364ULL:0.0013302'
            or plan.get('lua_sha256') != (
                '807e708e098654cfaf8cf58ebd91efc15f850d3da31795b02d28bc063eeb024b')
            or set((*SOURCE_FILES, *EXTRA_SOURCES))-set(plan.get('input_sha256', {}))
            or plan['input_sha256'].get(BASE_CHILD.replace('.', '/')+'.py') != BASE_SHA
            or not finite(plan.get('expected_initial_spline'))
            or not 0 <= plan['expected_initial_spline'] < .275
            or plan.get('observer_source') != 'native_graphics.normalizedCarPosition@248'
            or plan.get('native_graphics_prefix_bytes') != 264
            or plan.get('native_graphics_pair_minimum_seconds') != .08
            or plan.get('observation_timeout_seconds') != 2.):
        raise ValueError('Exact finite3 diagnostic/native-observer contract required')
    from flybrain_drive.session_heartbeat_evidence_v1 import require_local_c_output

    for key in ('output', 'preflight_output'):
        require_local_c_output(plan[key])
    if Path(plan['output']).resolve() == Path(plan['preflight_output']).resolve():
        raise ValueError('Preflight and actual outputs must be separate new directories')


def verify_inputs(plan, plan_file, original_plan_sha):
    if sha(plan_file) != original_plan_sha:
        raise ValueError('Original sealed plan changed; no repinning')
    for name, expected in plan['input_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Registered source/input changed; no retry: '+name)
    if shutil.disk_usage('C:/').free < plan['minimum_free_disk_bytes']:
        raise ValueError('At least8GiB free required before another complete task')


def check_idle(value, age_seconds, plan, window_handle, *, after_source=None):
    """Normal CSP status has no route; route is never read from this dictionary."""
    if (not finite(age_seconds) or not 0 <= age_seconds <= 2
            or value.get('generation') != plan['generation']
            or value.get('epoch') != plan['epoch']
            or value.get('window_handle') != str(window_handle)+'ULL'
            or value.get('car_id') != 'bmw_z4_gt3' or value.get('track_id') != 'monza'
            or value.get('paused') is not True or value.get('ai_controlled') is not False
            or value.get('is_online') is not False
            or any(value.get(k) for k in ('camera_active', 'expert_probe_active',
                'brake_settle_active', 'pending_receipt_id', 'error', 'in_menu'))
            or any(not finite(value.get(k)) for k in ('speed_kmh', 'gas', 'brake', 'script_clock'))
            or abs(value['speed_kmh']) >= .1 or not 0 <= value['gas'] < .02
            or not .98 < value['brake'] <= 1):
        raise ValueError('Fresh same-plant paused AIoff/fullbrake idle status required')
    return after_source is None or value['script_clock'] > after_source


def decode_observer(raw, received):
    from flybrain_drive.ac_native_clock import GRAPHICS_PREFIX, decode_graphics_timing
    from flybrain_drive.ac_native_position import decode_diagnostic_position

    position = decode_diagnostic_position(raw, received)
    timing = decode_graphics_timing(raw[:GRAPHICS_PREFIX.size], received)
    point = struct.unpack_from('<f', raw, 248)[0]
    if (position.status != 'pause' or timing.completed_laps != 0
            or not finite(point) or not 0 <= point < 1):
        raise ValueError('Paused lap0 native normalizedCarPosition observer required')
    return dict(spline_position=point, lap_count=timing.completed_laps,
        graphics_packet=position.packet_id, graphics_status=position.status,
        received_perf_counter_seconds=position.received_perf_counter_seconds,
        coordinates_native_xyz_m=list(position.coordinates_native_xyz_m),
        source='native_graphics.normalizedCarPosition@248', actor_input=False,
        exact_physical_sample_time_verified=False, repeated_copy_is_engine_seqlock=False)


def validate_pair(first, second):
    if (second['observer']['graphics_packet'] <= first['observer']['graphics_packet']
            or second['physics420_sha256'] != first['physics420_sha256']
            or second['native_packet'] != first['native_packet']
            or second['observer']['received_perf_counter_seconds'] <= (
                first['observer']['received_perf_counter_seconds'])
            or second['status']['script_clock'] < first['status']['script_clock']):
        raise ValueError('Two advancing graphics packets with unchanged paused physics required')


def stable_native_copy(mapping):
    for attempt in range(8):
        first, second = mapping.read(), mapping.read()
        if first == second:
            return second, attempt+1
    raise RuntimeError('Native bytes changed during every bounded repeated-copy attempt')


def coordinate_link(plan):
    """Saved no-motion coordinate agreement; not continuous track geometry calibration."""
    link = plan['coordinate_link']
    receipt = json.loads(Path(link['CSP_receipt']).read_bytes())
    guard = json.loads(Path(link['native_guard_summary']).read_bytes())
    csp = receipt['after']['diagnostic_route']['spline_position']
    native = guard['route_for_diagnostic_staging_only']['spline_position']
    if (receipt['after']['generation'] != plan['generation']
            or receipt['after']['epoch'] != plan['epoch']
            or receipt['after']['paused'] is not True
            or receipt['after']['diagnostic_route']['actor_input'] is not False
            or receipt['after']['diagnostic_route']['lap_count'] != 0
            or guard['current_engine_pid'] != plan['engine_pid']
            or guard['current_generation'] != plan['generation']
            or guard['route_for_diagnostic_staging_only']['lap_count'] != 0
            or guard['route_for_diagnostic_staging_only']['graphics_advance_observed'] is not True
            or not finite(csp) or not finite(native)
            or abs(csp-native) > 1e-7 or abs(native-plan['expected_initial_spline']) > 1e-7
            or link['no_motion_between_saved_observers_root_attestation'] is not True):
        raise ValueError('Saved CSP after/native paused coordinate applicability link differs')
    return dict(CSP_after_spline=csp, native_guard_spline=native,
        absolute_difference=abs(csp-native), source='saved1815after/saved1824native',
        no_motion_between_is_external_attestation=True, continuous_geometry_verified=False,
        actor_input=False, fit_allowed=False)


def observe_paused(plan, directory, *, after_source=None):
    """GET-only: existing mappings, status file and vJoy status; no activation/pings."""
    from flybrain_drive.ac_capture import WindowsACWindow
    from flybrain_drive.ac_image_buffer import ExistingNamedMapping
    from flybrain_drive.ac_native_physics import decode_static_identity_prefix
    from flybrain_drive.ac_published_state import read_published_json
    from flybrain_drive.ac_teacher_physics import decode_teacher_physics
    from flybrain_drive.ac_vjoy import load_vjoy_sdk
    from flybrain_drive.evidence import write_json
    from flybrain_drive.session_heartbeat_admission import refuse_leased_transport

    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    window_api = WindowsACWindow()
    windows = window_api.discover()
    if len(windows) != 1 or windows[0].pid != plan['engine_pid']:
        raise ValueError('One exact registered engine PID required')
    window = windows[0]
    if window_api.user.GetForegroundWindow() == window.hwnd:
        raise ValueError('Background engine required; never automate the window')
    installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha(installed) != plan['lua_sha256']:
        raise ValueError('Exact installed807 required; never deploy from staging')
    axis = json.loads(Path('configs/ac-axis-probe.json').read_bytes())
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    if sdk.GetVJDStatus(axis['device_id']) != 1:
        raise ValueError('FREE controller required; staging never acquires a driver')

    def idle():
        refuse_leased_transport('GET-only staging observer', status_path=status_path)
        value, modified = read_published_json(status_path)
        ready = check_idle(value, time.time()-modified, plan, window.hwnd,
                           after_source=after_source)
        return value, modified, ready

    deadline = time.perf_counter()+plan['observation_timeout_seconds']
    while True:
        _, _, ready = idle()
        if ready:
            break
        if time.perf_counter() >= deadline:
            raise TimeoutError('No advancing fresh SYNC after frozen handoff; no new owner')
        time.sleep(.01)
    with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
        static, _ = stable_native_copy(mapping)
    (directory/'static200.bin').write_bytes(static)
    if decode_static_identity_prefix(static) != {
            'shared_memory_version':'1.7','ac_version':'1.16.4',
            'car_model':'bmw_z4_gt3','track':'monza'}:
        raise ValueError('Exact repeated native static identity required')
    samples = []
    with (ExistingNamedMapping('Local\\acpmf_physics', 420) as physics,
          ExistingNamedMapping('Local\\acpmf_graphics', 264) as graphics):
        for index in range(2):
            physics_raw, physics_attempts = stable_native_copy(physics)
            physics_received = time.perf_counter()
            graphics_raw, graphics_attempts = stable_native_copy(graphics)
            graphics_copied = time.perf_counter()
            graphics_file = directory/('graphics-%d-264.bin' % index)
            physics_file = directory/('physics-AI-%d-420.bin' % index)
            graphics_file.write_bytes(graphics_raw)
            physics_file.write_bytes(physics_raw)
            # Original offending bytes and read receipts survive even decoder/guard failure.
            write_json(directory/('raw-read-%d.json' % index), dict(
                graphics_file=str(graphics_file), graphics_sha256=sha(graphics_file),
                graphics_copy_host_perf_counter_seconds=graphics_copied,
                graphics_repeated_copy_attempts=graphics_attempts,
                physics420_file=str(physics_file), physics420_sha256=sha(physics_file),
                physics_copy_host_perf_counter_seconds=physics_received,
                physics_repeated_copy_attempts=physics_attempts, actor_input=False))
            graphics_received = graphics_copied
            if samples and struct.unpack_from('<i', graphics_raw)[0] == (
                    samples[0]['observer']['graphics_packet']):
                if graphics_raw != Path(samples[0]['graphics_file']).read_bytes():
                    raise ValueError('Duplicate graphics packet changed bytes')
                graphics_received = samples[0]['observer']['received_perf_counter_seconds']
            observer = decode_observer(graphics_raw, graphics_received)
            coupled = decode_teacher_physics(physics_raw, physics_received)
            state, modified, ready = idle()
            if not ready:
                raise ValueError('SYNC clock did not retain advancement after handoff')
            f = coupled.physics.fields
            if abs(f['speedKmh']) >= .1 or not 0 <= f['gas'] < .02 or not .98 < f['brake'] <= 1:
                raise ValueError('Original native paused stationary fullbrake required')
            sample = dict(observer=observer, status=state, status_file_mtime=modified,
                status_original_bytes_claimed=False,
                graphics_file=str(graphics_file), graphics_sha256=sha(graphics_file),
                physics420_file=str(physics_file), physics420_sha256=sha(physics_file),
                native_packet=coupled.physics.packet_id,
                native_read_host_receipt=coupled.physics.received_monotonic_seconds,
                native_AI_frozen_baseline=coupled.builtin_ai_controlled,
                native_gear=struct.unpack_from('<i', coupled.physics.prefix_bytes, 16)[0],
                native_speed_kmh=f['speedKmh'], native_gas=f['gas'], native_brake=f['brake'],
                fresh_motion_or_AIownership_claimed=False, actor_input=False)
            samples.append(sample)
            write_json(directory/('sample-%d.json' % index), sample)
            if index == 0:
                time.sleep(plan['native_graphics_pair_minimum_seconds'])
    validate_pair(*samples)
    if (sdk.GetVJDStatus(axis['device_id']) != 1
            or window_api.user.GetForegroundWindow() == window.hwnd):
        raise ValueError('FREE/background must remain true through observer read')
    if not idle()[2]:
        raise ValueError('Advancing no-lease SYNC must remain valid through observer end')
    return dict(observer=samples[-1]['observer'], samples=samples,
        FREE_background=True, same_frozen_physics=True, graphics_advance_observed=True,
        no_fixed_gear_admission=True, driver_acquisitions=0, vehicle_API_calls=0,
        CSP_clock_pings=0, actor_input=False)


def transit_decision(point):
    if not finite(point) or not 0 <= point < 1:
        raise ValueError('Finite original native normalized route required')
    if .275 <= point <= .44:
        return 'stop_reached'
    if point > .44:
        raise ValueError('Target passed; never start another lap or enlarge TRAIN geography')
    return 'continue_bounded_task'


def check_child(value):
    handoff = value.get('handoff', {})
    if (value.get('status') != 'completed_diagnostic_transport_collection'
            or value.get('role') != 'diagnostic' or value.get('fit_allowed') is not False
            or value.get('heartbeat_evidence', {}).get('status') != 'complete'
            or value.get('transport_state') != 'closed' or value.get('transport_error') is not None
            or value.get('owned_threads_still_live') != []
            or value.get('lease_retained') is not False
            or handoff.get('same_writer_joined') is not True
            or handoff.get('lease_removed') is not True
            or not finite(handoff.get('original_source_clock'))):
        raise ValueError('Completed exact diagnostic child/normal handoff/evidence required; stop')
    return handoff['original_source_clock']


def execute_stages(plan, result, *, observe, child, verify, checkpoint):
    """Sequential finite orchestration; tests inject only pure saved-fixture callbacks."""
    state = observe(None, 0)
    if abs(state['observer']['spline_position']-plan['expected_initial_spline']) > 1e-4:
        raise ValueError('Fresh native start differs from registered no-motion baseline')
    for index in range(3):
        point = state['observer']['spline_position']
        if transit_decision(point) == 'stop_reached':
            return dict(status='completed_diagnostic_transit_reached_registered_start',
                        final_spline=point)
        verify()
        record = dict(index=index, role='transit', fit_allowed=False, status='started',
                      before_spline=point, before_observer=state)
        result['attempts'].append(record)
        checkpoint()
        actual, actual_sha = child(index)
        source = check_child(actual)
        record.update(status='completed_qualified_diagnostic_task', child_result_sha256=actual_sha,
                      original_frozen_handoff_clock=source)
        checkpoint()
        state = observe(source, index+1)
        current = state['observer']['spline_position']
        record.update(after_spline=current, after_observer=state)
        checkpoint()
        if current < point-.001:
            raise ValueError('Native route reversed/wrapped; stop without another activation')
        transit_decision(current)  # Also fail on overshoot after final permitted child.
    point = state['observer']['spline_position']
    reached = transit_decision(point) == 'stop_reached'
    return dict(status=('completed_diagnostic_transit_reached_registered_start' if reached
        else 'completed_finite_transit_start_not_reached'), final_spline=point)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',
        default='configs/ac-diagnostic-transit-to-train-start-v2-20261030.json')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight-only', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-result-sha256')
    args = parser.parse_args()
    plan_file = Path(args.plan)
    plan = json.loads(plan_file.read_bytes())
    validate_plan(plan)
    if not args.preflight_only and not args.execute:
        print(json.dumps(dict(mode='inventory_only', arguments=vars(args), live_actions=0)))
        return
    plan_sha = sha(plan_file)
    verify_inputs(plan, plan_file, plan_sha)
    link = coordinate_link(plan)
    if args.execute:
        preflight = Path(plan['preflight_output'])/'result.json'
        if not args.preflight_result_sha256 or sha(preflight) != args.preflight_result_sha256:
            raise ValueError('Explicit exact successful read-only preflight result SHA required')
        proof = json.loads(preflight.read_bytes())
        if (proof.get('status') != 'completed_read_only_preflight'
                or proof.get('plan_sha256') != plan_sha
                or proof.get('input_sha256') != plan['input_sha256']
                or proof.get('transit_activations') != 0):
            raise ValueError('Read-only preflight must bind this exact sealed plan/sources')
    from flybrain_drive.evidence import begin_run, write_json

    output = begin_run(plan['preflight_output'] if args.preflight_only else plan['output'],
        dict(role='transit', fit_allowed=False,
             mode='preflight_only' if args.preflight_only else 'transit',
             plan_sha256=plan_sha, program_sha256=sha(__file__),
             declaration=plan, arguments=vars(args)))
    (output/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', role='transit', fit_allowed=False, optimizer_updates=0,
        student_deployed=False, goal_achieved=False, attempts=[], transit_activations=0,
        maximum_activations=3, plan_sha256=plan_sha, input_sha256=plan['input_sha256'],
        coordinate_link=link, preflight_only=args.preflight_only,
        game_resets=0, deployments=0, UI_actions=0, GPU_actions=0)

    def observe(after_source, index):
        return observe_paused(plan, output/('observer-%02d' % index), after_source=after_source)

    def verify():
        verify_inputs(plan, plan_file, plan_sha)

    def child(index):
        attempt = output/('attempt-%02d' % index)
        original_argv = sys.argv
        try:
            sys.argv = [BASE_CHILD, '--mode', 'pace30',
                '--expected-engine-pid', str(plan['engine_pid']),
                '--expected-generation', plan['generation'], '--expected-epoch', str(plan['epoch']),
                '--expected-installed-sha256', plan['lua_sha256'], '--frame-persistence-profile',
                'configs/ac-frame-persistence-buffered32-20261030.json',
                '--output', str(attempt), '--execute']
            result['transit_activations'] += 1
            runpy.run_module(BASE_CHILD, run_name='__main__')
        finally:
            sys.argv = original_argv
        return json.loads((attempt/'result.json').read_bytes()), sha(attempt/'result.json')

    try:
        if args.preflight_only:
            result['baseline'] = observe(None, 0)
            point = result['baseline']['observer']['spline_position']
            if abs(point-plan['expected_initial_spline']) > 1e-4:
                raise ValueError('Read-only native start differs from registered baseline')
            verify()
            result['status'] = 'completed_read_only_preflight'
        else:
            result.update(execute_stages(plan, result, observe=observe, child=child, verify=verify,
                checkpoint=lambda: write_json(output/'partial-result.json', result)))
            verify()
    except BaseException as error:
        result.update(status='failed', error=type(error).__name__, reason=str(error))
        raise
    finally:
        result['program_sha256'] = sha(__file__)
        write_json(output/'result.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in (
            'attempts','baseline','input_sha256')}), flush=True)


if __name__ == '__main__':
    main()
