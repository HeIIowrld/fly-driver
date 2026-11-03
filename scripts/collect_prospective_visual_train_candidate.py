"""Collect one predeclared geographic TRAIN candidate; fit is never granted here."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_transit_observer_v1 import observe_paused
from flybrain_drive.spatial_teacher_plan import next_lease, validate_lease_route


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan, plan_hash = read(plan_path), sha256(plan_path)
    if (plan['schema'] != 'flybrain.visual_train_candidate.v1'
            or plan['role'] != 'train_candidate' or plan['fit_allowed'] is not False
            or plan['automatic_retry'] is not False
            or plan['spatial_config_sha256'] != sha256(plan['spatial_config'])
            or plan['camera_config_sha256'] != sha256(plan['camera_config'])
            or plan['recorder_sha256'] != sha256('scripts/record_ac_engine_ai_probe.py')
            or plan['observer_sha256'] != sha256('src/flybrain_drive/native_transit_observer_v1.py')
            or plan['collector_sha256'] != sha256(__file__)):
        raise ValueError('Exact predeclared non-fit candidate and original sources required')
    required = ('scripts/control_ac_session.py', 'src/flybrain_drive/ac_csp_vision.py',
                'src/flybrain_drive/ac_csp_retina.py', 'src/flybrain_drive/ac_shared_texture.py',
                'src/flybrain_drive/ac_native_physics.py',
                'src/flybrain_drive/ac_teacher_physics.py',
                'src/flybrain_drive/spatial_teacher_plan.py',
                'integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    if (set(plan['source_sha256']) != set(required)
            or any(sha256(path) != plan['source_sha256'][path] for path in required)):
        raise ValueError('Exact current sensor, control and route source bindings required')
    previous = Path(plan['previous_observation'])
    if sha256(previous / 'result.json') != plan['previous_observation_sha256']:
        raise ValueError('Original position observation changed')
    previous_result = read(previous / 'result.json')
    if (previous_result['status'] != 'completed'
            or abs(previous_result['observer']['spline_position'] - plan['expected_start']) > 1e-6):
        raise ValueError('Exact earlier paused position evidence required')
    spatial = read(plan['spatial_config'])
    prospective = next_lease(plan['expected_start'], spatial['regions'])
    if (prospective['role'] != 'train' or prospective['duration_seconds'] != 30
            or prospective['region'] != {'role': 'train', 'start': .27, 'end': .63}):
        raise ValueError('Original declared train region and full lease required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'plan': plan,
        'plan_sha256': plan_hash, 'prospective_lease': prospective,
        'role': 'train_candidate', 'fit_allowed': False, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run / 'declared-plan.json').write_bytes(plan_path.read_bytes())
    result = {'status': 'failed', 'role': 'train_candidate', 'fit_allowed': False,
              'actor_deployed': False, 'optimizer_updates': 0}
    try:
        observation = observe_paused(plan['observer_plan'], run / 'observer')
        if abs(observation['observer']['spline_position'] - plan['expected_start']) > 1e-4:
            raise ValueError('Fresh actual start no longer matches predeclared position')
        write_json(run / 'before-observation.json', observation)
        if sha256(plan_path) != plan_hash:
            raise ValueError('Predeclared plan changed')
        command = [sys.executable, '-m', 'scripts.record_ac_engine_ai_probe',
            '--config', plan['camera_config'], '--image-format', 'bmp', '--shared-transport',
            '--batched-shared-copy', '--async-pixel-precompute', '--async-frame-persistence',
            '--mmf-metadata', '--source-timing', '--same-copy-native-authority',
            '--duration-seconds', '30', '--speed-hint-kmh', '60',
            '--ai-steer-multiplier', '.9', '--output', str(run / 'recording')]
        with (run / 'child.log').open('x', encoding='utf8') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            result['child_pid'] = child.pid
            write_json(run / 'progress.json', result)
            try:
                exit_code = child.wait(timeout=90)
            except subprocess.TimeoutExpired as error:
                result.update(status='child_outcome_unknown', child_still_live=True)
                raise TimeoutError('Original owned child still live; no retry') from error
        result['child_exit_code'] = exit_code
        child_result = read(run / 'recording/result.json')
        result['recording_result_sha256'] = sha256(run / 'recording/result.json')
        if exit_code or child_result['status'] != 'completed':
            raise ValueError('Original recording failed; preserve whole history')
        receipt_path = run / 'recording/control/api-receipt.json'
        receipt = read(receipt_path)
        route = validate_lease_route(prospective, receipt)
        result.update(status='completed_prospective_train_candidate',
            route_diagnostic=route, receipt_sha256=sha256(receipt_path),
            raw_audit_required=True, time_alignment_required=True,
            teacher_label_verification_required=True, fit_allowed=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
