"""Prepare one stationary AC episode through API/vJoy; no desktop automation or retries."""

import argparse
import json
import subprocess
import sys
import time

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, NativePhysicsReader
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_wgc import physics_record


def stationary_sample():
    deadline = time.monotonic()+3
    stable = 0
    with NativePhysicsReader(timeout=.15) as reader:
        while time.monotonic() < deadline:
            sample = reader.receive()
            if (abs(sample.fields['speedKmh']) < .1 and sample.fields['gas'] < .02
                    and sample.fields['brake'] > .98):
                stable += 1
                if stable >= 2:
                    return sample
            else:
                stable = 0
    raise TimeoutError('Native car did not settle into stationary brake hold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preserve-position', action='store_true',
                        help='Prepare a stationary continuation without resetting to the spawn')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862833,
                                     ui_actions_sent=0, retry_policy='no automatic retries'))
    result = dict(status='failed', stages=[], ui_actions_sent=0, goal_achieved=False)

    def stage(name, module, options):
        destination = run / name
        subprocess.run([sys.executable, '-m', module, *options,
                        '--output', str(destination)], check=True)
        value = json.loads((destination / 'result.json').read_text(encoding='utf8'))
        result['stages'].append(dict(name=name, status=value['status']))
        if value['status'] not in ('completed', 'read_only'):
            raise RuntimeError('Preparation stage failed: '+name)
        return value

    try:
        status = stage('status', 'scripts.control_ac_session', ['--command', 'status'])
        if not args.execute:
            result.update(status='read_only', session=status['session'])
        else:
            if status['session']['paused']:
                stage('resume', 'scripts.control_ac_session',
                      ['--command', 'resume', '--execute'])
            if not args.preserve_position:
                stage('restart', 'scripts.control_ac_session',
                      ['--command', 'restart', '--execute'])
            write_json(run / 'settled-before-start.json', physics_record(stationary_sample()))
            if args.preserve_position:
                current = stage('continuation-status', 'scripts.control_ac_session',
                                ['--command', 'status'])['session']
                if current['paused']:
                    raise ValueError('Continuation unexpectedly paused after preparation')
                if current['in_menu']:
                    stage('start', 'scripts.control_ac_session',
                          ['--command', 'start', '--execute'])
                elif not current['live']:
                    raise ValueError('Continuation is neither live nor at the start menu')
            else:
                stage('start', 'scripts.control_ac_session', ['--command', 'start', '--execute'])
            write_json(run / 'settled-after-start.json', physics_record(stationary_sample()))
            calibration = stage('calibration', 'scripts.calibrate_ac_sdk_steering',
                                ['--amplitude', '.6'])
            calibration_path = run / 'calibration/result.json'
            sample = stationary_sample()
            gear = PHYSICS_PREFIX.unpack(sample.prefix_bytes)[4]
            if gear == 1:
                engaged = stage('first-gear', 'scripts.probe_ac_automatic_launch',
                    ['--execute', '--gear-up-button', '0',
                     '--calibration-result', str(calibration_path)])
                if not engaged['first_gear_observed']:
                    raise ValueError('First gear was not observed')
            elif gear != 2:
                raise ValueError('Unexpected native gear during episode preparation')
            final = stationary_sample()
            if PHYSICS_PREFIX.unpack(final.prefix_bytes)[4] != 2:
                raise ValueError('Final native first gear check failed')
            write_json(run / 'ready-native.json', physics_record(final))
            result.update(status='completed', ready=True,
                position_preserved=args.preserve_position,
                calibration_result=str(calibration_path),
                producer_session_id=calibration['producer_session_id'],
                scope='stationary first gear, brake held; no model driving commands')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run / 'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
