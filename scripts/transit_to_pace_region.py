"""Finite transit-only old60kmh leases; paused diagnostic geography never becomes fit data."""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.collect_spatial_flight_teacher import idle
from scripts.control_ac_session import validate_probe_receipt

GENERATION = '37294426ULL:0.0014905'
ENGINE_PID = 45604
ELIGIBLE = (.275, .44)
PREFERRED = (.285, .425)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def transit_decision(spline, *, approaching=False, wraps=0, previous=None,
                     eligible=ELIGIBLE, preferred=PREFERRED):
    """Duration is a planning estimate, not a physical arc-length bound."""
    if (not math.isfinite(spline) or not 0 <= spline < 1
            or wraps not in (0, 1) or previous is not None
            and (not math.isfinite(previous) or not 0 <= previous < 1)):
        raise ValueError('Finite paused diagnostic spline and at most one initial wrap required')
    if previous is not None and previous-spline > .5:
        if approaching or wraps:
            raise ValueError('Target passed or second wrap; never start another lap')
        wraps += 1
        approaching = True
    approaching = approaching or spline < eligible[0]
    if approaching and spline > eligible[1]:
        raise ValueError('Approached target was passed above .44; no another-lap retry')
    ready = eligible[0] <= spline <= eligible[1]
    distance = (eligible[0]-spline) % 1
    return dict(action='stop' if ready else 'transit', role='transit', fit_allowed=False,
        spline=spline, preferred_start=preferred[0] <= spline <= preferred[1],
        duration_seconds=None if ready else 30 if distance > .06 else 3,
        forward_distance_planning_only=distance, approaching=approaching, wraps=wraps,
        estimate_is_not_physical_arc_bound=True, actor_input=False)


def paused_spline():
    with ExistingNamedMapping('Local\\acpmf_graphics', 264) as mapping:
        first, second = mapping.read(), mapping.read()
    if first != second or struct.unpack_from('<i', first, 4)[0] != 3:
        raise ValueError('Consistent paused native graphics baseline required')
    spline = struct.unpack_from('<f', first, 248)[0]
    if not math.isfinite(spline) or not 0 <= spline < 1:
        raise ValueError('Finite paused native diagnostic route required')
    return dict(spline_position=spline, actor_input=False,
        source='paused native graphics baseline; not fresh motion or student ownership')


def verify_sources(plan, plan_path, expected_plan_hash):
    if sha256(plan_path) != expected_plan_hash:
        raise ValueError('Presealed transit plan changed')
    for source, digest in plan['input_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Transit source/input changed: '+source)


def validate_plan(plan):
    v2 = plan['schema'] in ('flybrain.old60_transit_to_pace_region.v2',
                            'flybrain.old60_transit_to_pace_region.v3')
    v3 = plan['schema'] == 'flybrain.old60_transit_to_pace_region.v3'
    if (plan['schema'] not in ('flybrain.old60_transit_to_pace_region.v1',
                              'flybrain.old60_transit_to_pace_region.v2',
                              'flybrain.old60_transit_to_pace_region.v3')
            or plan['role'] != 'transit' or plan['fit_allowed'] is not False
            or plan['role_reassignment_allowed'] is not False
            or plan['actor_world_input'] is not False
            or not isinstance(plan['generation'], str) or not plan['generation']
            or type(plan['engine_pid']) is not int or plan['engine_pid'] <= 0
            or not v2 and (plan['generation'] != GENERATION or plan['engine_pid'] != ENGINE_PID)
            or v2 and plan.get('stop_camera_after_owned_ai') is not True
            or not 1 <= plan['maximum_stages'] <= 12
            or plan['eligible_start_interval'] != ([.67, .72] if v3 else list(ELIGIBLE))
            or plan['preferred_start_interval'] != ([.68, .71] if v3 else list(PREFERRED))
            or plan['long_if_forward_distance_exceeds'] != .06
            or plan['teacher_speed_kmh'] != 60 or plan['ai_steer_multiplier'] != .9
            or plan['minimum_free_bytes'] != 40_000_000_000
            or plan['child_duration_seconds'] != [3, 30]
            or plan['wrapper_timeouts_or_pid_kills'] is not False):
        raise ValueError('Exact finite transit-only existing60kmh profile required')
    baseline = read(plan['paused_baseline_result'])
    if (baseline['status'] != 'completed_read_only_paused_collection_baseline'
            or baseline['generation'] != plan['generation']
            or baseline['engine_pid'] != plan['engine_pid']
            or baseline['actor_input'] is not False):
        raise ValueError('Pinned planning baseline belongs to another engine/generation')
    if v2 and plan['input_sha256'].get(plan['paused_baseline_result']) != sha256(
            plan['paused_baseline_result']):
        raise ValueError('Current generation baseline must be pinned')
    for name in ('raw_camera_qualification_result', 'pace_raw_audit_result'):
        proof = read(plan[name])
        if (proof['status'] != 'completed' or proof['intervals_above_80ms']
                or any(proof['actor_replay_max_errors'].values())
                or proof['camera_aid_state_verified'] is not True
                or proof['raw_mmf_slots_and_original_metadata_verified'] is not True
                or proof['preactivation_baseline_advance_verified'] is not True):
            raise ValueError('Pinned complete raw camera qualification required')


def child_command(plan, output, duration):
    if duration not in (3, 30):
        raise ValueError('Only existing3/30second60kmh finite profiles are allowed')
    command = [sys.executable, '-m', 'scripts.record_ac_engine_ai_probe', '--config',
        plan['camera_configuration'], '--image-format', 'bmp', '--shared-transport',
        '--batched-shared-copy', '--async-pixel-precompute', '--async-frame-persistence',
        '--mmf-metadata', '--duration-seconds', str(duration), '--speed-hint-kmh', '60',
        '--ai-steer-multiplier', '.9', '--output', str(output)]
    if plan.get('stop_camera_after_owned_ai', False):
        command.append('--stop-camera-after-owned-ai')
    return command


def check_child(recording, control, receipt, duration, *, generation=GENERATION):
    expected = 'expert-motion-speed60' if duration == 3 else 'expert-collect30-speed60'
    if (recording['status'] != 'completed' or recording['vision']['error']
            or recording.get('control_process_still_live')
            or control['status'] != 'completed' or control.get('target_state_observed') is not True
            or control.get('lifecycle_requests_sent') != 1
            or control.get('foreground_before') is not False
            or control.get('foreground_after') is not False
            or receipt['status'] != 'api_returned_true' or receipt['command'] != expected
            or receipt['before']['generation'] != generation
            or receipt['after']['generation'] != generation):
        raise ValueError('Finite child/control/receipt failed; preserve failure and never retry')
    validate_probe_receipt(receipt, motion=True, collection=duration == 30,
                           fast=True, speed60=True)


def snapshot_sources(run, plan):
    project = Path.cwd().resolve()
    for index, (name, digest) in enumerate(plan['input_sha256'].items()):
        path = Path(name).resolve()
        if path.suffix not in ('.py', '.json', '.lua'):
            raise ValueError('Only declared source/JSON metadata may be snapshotted')
        relative = path.relative_to(project) if path.is_relative_to(project) else (
            Path('external-metadata')/f'{index:03d}-{path.name}')
        target = run/'executed-inputs'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if sha256(target) != digest:
            raise ValueError('Source changed while snapshotting before activation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan, plan_hash = read(args.plan), sha256(args.plan)
    validate_plan(plan)
    verify_sources(plan, args.plan, plan_hash)
    if not args.execute:
        print(json.dumps(dict(status='validated_ready_without_runtime_access',
            role='transit', fit_allowed=False, runtime_mappings_opened=0,
            control_requests_sent=0, maximum_stages=plan['maximum_stages'])))
        return
    project = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != project:
        raise ValueError('Execute only from the actual existing flybrain project root')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=plan_hash,
        program_sha256=sha256(__file__), role='transit', fit_allowed=False, scope=__doc__))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'transit-plan.json').write_bytes(Path(args.plan).read_bytes())
    result = dict(status='failed', role='transit', fit_allowed=False,
        role_reassignment_allowed=False, episodes=[], ui_actions_sent=0,
        student_deployed=False, optimizer_updates=0, actor_world_input=False,
        original_payloads_copied=0, wrapper_timeouts_or_pid_kills=False, goal_achieved=False)
    child, approaching, wraps, previous = None, False, 0, None
    try:
        snapshot_sources(run, plan)
        for index in range(plan['maximum_stages']+1):
            verify_sources(plan, args.plan, plan_hash)
            if shutil.disk_usage(os.environ['LOCALAPPDATA']).free < plan['minimum_free_bytes']:
                raise ValueError('Less than40GB free before finite transit stage')
            before = idle(plan['generation'])
            if before['engine_pid'] != plan['engine_pid']:
                raise ValueError('Fresh idle engine PID differs')
            position = paused_spline()
            decision = transit_decision(position['spline_position'], approaching=approaching,
                wraps=wraps, previous=previous, eligible=plan['eligible_start_interval'],
                preferred=plan['preferred_start_interval'])
            approaching, wraps = decision['approaching'], decision['wraps']
            if not result.get('initial_position'):
                result['initial_position'] = position
            result['last_position'] = position
            if decision['action'] == 'stop':
                result.update(status='completed_at_eligible_paused_start',
                    preferred_start=decision['preferred_start'], wraps=wraps,
                    no_new_activation_at_eligible_position=True)
                break
            if index == plan['maximum_stages']:
                raise ValueError('Finite transit stage budget exhausted; never start another batch')
            name = f'transit-{index:02d}'
            output = run/name
            command = child_command(plan, output, decision['duration_seconds'])
            write_json(run/(name+'-assignment.json'), dict(decision=decision, before=before,
                diagnostic_position=position, role='transit', fit_allowed=False,
                command=command, plan_sha256=plan_hash))
            with (run/(name+'.log')).open('x', encoding='utf8') as log:
                verify_sources(plan, args.plan, plan_hash)
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(project), creationflags=subprocess.CREATE_NO_WINDOW)
                print(json.dumps(dict(stage=name, pid=child.pid, role='transit',
                    fit_allowed=False, duration_seconds=decision['duration_seconds'])), flush=True)
                write_json(run/'live-child.json', dict(stage=name, pid=child.pid,
                    command=command, active=True, role='transit', fit_allowed=False,
                    created_no_window=True, no_timeout_or_restart=True))
                code = child.wait()
            write_json(run/'live-child.json', dict(stage=name, pid=child.pid,
                active=False, terminal_exit_code=code, role='transit', fit_allowed=False))
            entry = dict(index=index, path=str(output), role='transit', fit_allowed=False,
                duration_seconds=decision['duration_seconds'], child_pid=child.pid,
                child_exit_code=code, before_position=position)
            result['episodes'].append(entry)
            if code:
                raise ValueError('Finite recording child exited nonzero; no retry')
            verify_sources(plan, args.plan, plan_hash)
            recording = read(output/'result.json')
            control = read(output/'control/result.json')
            receipt = read(output/'control/api-receipt.json')
            check_child(recording, control, receipt, decision['duration_seconds'],
                        generation=plan['generation'])
            after = idle(plan['generation'])
            if after['engine_pid'] != plan['engine_pid']:
                raise ValueError('Post-stage engine PID differs')
            after_position = paused_spline()
            entry.update(after=after, after_position=after_position,
                result_sha256=sha256(output/'result.json'),
                control_result_sha256=sha256(output/'control/result.json'),
                receipt_sha256=sha256(output/'control/api-receipt.json'),
                decisions=recording['decisions'],
                camera_wait_misses=recording['camera_wait_misses'])
            result['last_position'] = after_position
            write_json(run/'partial-result.json', result)
            # Recheck the actual end position now; preserve this whole transit
            # even when it wrapped, overshot, or had cleanup observations.
            transit_decision(after_position['spline_position'], approaching=approaching,
                wraps=wraps, previous=position['spline_position'],
                eligible=plan['eligible_start_interval'], preferred=plan['preferred_start_interval'])
            previous = position['spline_position']
    except Exception as error:
        result.update(status='failed', error=type(error).__name__, reason=str(error))
    finally:
        if child is not None and child.poll() is None:
            # Observe the same finite owned child, never kill, restart or send
            # another control request because an observation failed.
            result['observed_same_live_child_to_terminal'] = child.pid
            result['last_child_terminal_exit_code'] = child.wait()
        write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result.get('reason', 'Transit failed'))


if __name__ == '__main__':
    main()
