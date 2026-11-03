"""Bounded API resume/native AI-flag observation/pause; no actuator or desktop input."""

import argparse
import json
import subprocess
import sys
import time

from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--new-episode', action='store_true',
                        help='Explicit API restart/start before observation; no neural state')
    args = parser.parse_args()
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args)))
    result = dict(status='failed', actuator_messages_sent=0, ui_actions_sent=0)
    reader, ai = NativePhysicsReader(timeout=.15), NativeTeacherStatusReader()
    samples = []

    def save(label):
        sample = reader.receive()
        row = dict(phase=label, native=physics_record(sample), ai=ai.snapshot())
        samples.append(row)
        write_json(run / 'samples.json', samples)
        f = sample.fields
        if abs(f['speedKmh']) >= .1 or f['gas'] >= .02 or f['brake'] <= .98:
            raise ValueError('Native stationary brake hold violated')

    def command(name):
        with (run / (name+'.log')).open('x', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', 'scripts.control_ac_session',
                '--command', name, '--execute', '--output', str(run / name)],
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            # Observe this same client to termination; no timeout-based replacement.
            child.wait()
        if child.returncode:
            raise RuntimeError('Lifecycle failed: '+name)

    try:
        save('paused_before')
        command('resume')
        if args.new_episode:
            command('restart')
            command('start')
        for _ in range(10):
            save('live')
            time.sleep(.02)
        result['status'] = 'observed'
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        try:
            command('pause')
            save('paused_after')
            result['pause_verified'] = True
        except Exception as error:
            result.update(status='failed', cleanup_error=str(error))
        reader.close()
        ai.close()
        write_json(run / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
