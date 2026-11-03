"""Prospectively assigned finite pace recording; raw/labels still require independent audits."""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.prospective_corpus import ROLES, admit_whole_history, validate_pace30_profile
from flybrain_drive.teacher_pace import PACE_COMMAND, validate_pace_receipt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan['schema'] != 'flybrain.prospective_pace_recording.v1'
            or plan['role'] not in ('diagnostic', 'development', *ROLES)
            or plan['fit_allowed'] is not False
            or plan['command'] != PACE_COMMAND or plan['actor_world_input'] is not False):
        raise ValueError('Preassigned whole-history pace role; no automatic fit admission')
    validate_pace30_profile(plan['profile'])
    native_history = plan.get('native_copy_history', False)
    camera_stop_after_ai = plan.get('stop_camera_after_owned_ai', False)
    same_copy_authority = plan.get('same_copy_native_authority', False)
    if type(native_history) is not bool:
        raise ValueError('Native history opt-in must be an explicit boolean')
    if type(camera_stop_after_ai) is not bool:
        raise ValueError('Camera lifecycle opt-in must be an explicit boolean')
    if type(same_copy_authority) is not bool:
        raise ValueError('Native same-copy authority opt-in must be an explicit boolean')
    if native_history or same_copy_authority:
        for source in ('scripts/record_ac_engine_ai_probe.py',
                       'scripts/record_ac_airflow_timed_diagnostic.py',
                       'src/flybrain_drive/ac_teacher_physics.py'):
            if source not in plan['input_sha256']:
                raise ValueError('Native history collection source must be pinned: '+source)
    for name, digest in plan['input_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Preassigned source/input changed: '+name)
    calibration = read(Path(plan['raw_camera_qualification'])/'result.json')
    if (calibration['status'] != 'completed' or calibration['intervals_above_80ms']
            or any(calibration['actor_replay_max_errors'].values())
            or calibration['camera_aid_state_verified'] is not True
            or calibration['preactivation_baseline_advance_verified'] is not True):
        raise ValueError('Same-source whole fresh camera qualification required')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), ui_actions_sent=0, student_deployed=False,
        actor_input=False, fit_allowed=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'executed-recorder.py').write_bytes(Path('scripts/record_ac_engine_ai_probe.py').read_bytes())
    result = dict(status='failed', role=plan['role'], fit_allowed=False,
        student_deployed=False, goal_achieved=False, ui_actions_sent=0,
        vjoy_messages_sent=0, optimizer_updates=0)
    try:
        project = Path.cwd().resolve()
        for name, digest in plan['input_sha256'].items():
            path = Path(name).resolve()
            if path.is_relative_to(project) and path.suffix in ('.py', '.json', '.lua'):
                target = run/'executed-inputs'/path.relative_to(project)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                if sha256(target) != digest:
                    raise ValueError('Execution input changed while snapshotting: '+name)
        state, modified = read_published_json(
            Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        windows = WindowsACWindow().discover()
        if (not 0 <= time.time()-modified <= 2 or state['generation'] != plan['generation']
                or state['paused'] is not True or state['ai_controlled'] is not False
                or state['camera_active'] or state['expert_probe_active'] or state.get('error')
                or len(windows) != 1 or windows[0].pid != plan['engine_pid']):
            raise ValueError('Same paused idle source required before camera/control child')
        with ExistingNamedMapping('Local\\acpmf_graphics', 264) as mapping:
            raw, second = mapping.read(), mapping.read()
        if raw != second:
            raise ValueError('Paused diagnostic route baseline changed')
        spline = struct.unpack_from('<f', raw, 248)[0]
        assignment = dict(role=plan['role'], expected_start_spline=spline,
            spline_interval=plan.get('spline_interval'), actor_input=False,
            source='paused diagnostic baseline; never fresh student ownership evidence')
        if plan['role'] != 'diagnostic':
            lo, hi = plan['eligible_start_interval']
            if not lo <= spline <= hi:
                raise ValueError('Start outside prospectively assigned whole-history region')
        write_json(run/'assignment-before-activation.json', assignment)
        command = [sys.executable, '-m', 'scripts.record_ac_engine_ai_probe',
            '--config', plan['camera_configuration'], '--image-format', 'bmp',
            '--shared-transport', '--batched-shared-copy', '--async-pixel-precompute',
            '--async-frame-persistence', '--mmf-metadata', '--duration-seconds', '30',
            '--speed-hint-kmh', '100', '--pace-task', '--ai-steer-multiplier', '.9',
            '--output', str(run/'recording')]
        if native_history:
            command.append('--native-copy-history')
        if camera_stop_after_ai:
            command.append('--stop-camera-after-owned-ai')
        if same_copy_authority:
            command.append('--same-copy-native-authority')
        with (run/'recording.log').open('x', encoding='utf8') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(run/'live-child.json', dict(pid=child.pid, command=command,
                role=plan['role'], schema='owned finite recording; never restart on timeout'))
            print(json.dumps(dict(pid=child.pid, role=plan['role'])), flush=True)
            code = child.wait()
        recorded = read(run/'recording/result.json')
        result.update(child_exit_code=code, recording_result_sha256=sha256(
            run/'recording/result.json'), recording_status=recorded['status'])
        if code or recorded['status'] != 'completed':
            raise ValueError('Owned pace recording failed; preserve original raw failure')
        receipt = read(run/'recording/control/api-receipt.json')
        result['pace_response'] = validate_pace_receipt(receipt)
        if plan['role'] != 'diagnostic':
            snapshots = [receipt['before'], *receipt['telemetry'], receipt['after']]
            route = [dict(s['diagnostic_route'], wheels_outside=s['wheels_outside'])
                     for s in snapshots]
            result['route_admission'] = admit_whole_history(assignment, route)
        result.update(status='completed_preassigned_pace_recording',
            decisions=recorded['decisions'], camera_wait_misses=recorded['camera_wait_misses'],
            raw_and_task_label_audits_required=True, role_reassignment_allowed=False,
            actual_new_condition_not_student_policy=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
