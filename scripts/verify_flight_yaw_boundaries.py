"""Separate fresh processes certify full and zero-yaw original-episode inference."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.owned_worker_wait import wait_owned_worker
from scripts.record_ac_engine_ai_probe import verified_shadow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--site-packages', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scratch-warmup', action='store_true')
    p.add_argument('--frozen-visual-tail', action='store_true')
    p.add_argument('--graphed-visual-tail', action='store_true')
    p.add_argument('--linear-yaw', action='store_true')
    a = p.parse_args()
    assert not (a.frozen_visual_tail and a.graphed_visual_tail)
    plan = read(Path(a.recording) / 'manifest.json')['config']['plan']
    _, proof = verified_shadow(plan['inference_config'], plan['inference_verification'])
    run = begin_run(a.output, dict(arguments=vars(a), base_verification=proof,
                                  actuator_messages_sent=0))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0)
    try:
        errors = {}
        bootstrap = ('import sys,site,runpy; p=sys.argv.pop(1); site.addsitedir(p); '
                     'sys.path.insert(0,p); '
                     'runpy.run_module("scripts.verify_flight_yaw_trial",run_name="__main__")')
        for mode in ('full', 'zero_yaw'):
            with (run / (mode+'.log')).open('x', encoding='utf8') as log:
                child = subprocess.Popen([sys.executable, '-c', bootstrap, a.site_packages,
                    '--recording', a.recording, '--audit', a.audit, '--mode', mode,
                    '--output', str(run / mode),
                    *(['--scratch-warmup'] if a.scratch_warmup else []),
                    *(['--frozen-visual-tail'] if a.frozen_visual_tail else []),
                    *(['--graphed-visual-tail'] if a.graphed_visual_tail else []),
                    *(['--linear-yaw'] if a.linear_yaw else [])],
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run / 'worker.json', dict(pid=child.pid, mode=mode))
                wait_owned_worker(child, module='scripts.verify_flight_yaw_trial',
                    result_file=run/mode/'result.json', receipt_file=run/(mode+'-process.json'))
            receipt = read(run / mode / 'result.json')
            assert child.returncode == 0 and receipt['status'] == 'completed_one_yaw_boundary'
            assert receipt['observations_per_mode'] == 225
            errors[mode] = receipt['maximum_intent_errors'][mode]
        result.update(status='completed_both_yaw_boundaries', observations_per_mode=225,
            scratch_warmup_verified=a.scratch_warmup,
            frozen_visual_tail_verified=a.frozen_visual_tail,
            graphed_visual_tail_verified=a.graphed_visual_tail,
            linear_yaw_verified=a.linear_yaw,
            maximum_intent_errors=errors, separate_fresh_processes=True,
            full_receipt_sha256=sha256(run / 'full/result.json'),
            zero_yaw_receipt_sha256=sha256(run / 'zero_yaw/result.json'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
