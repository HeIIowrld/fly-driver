"""One predeclared balanced four-trial physical block, fresh API episodes, no desktop input."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from flybrain_drive.ac_diagnostic_admission import verified_diagnostic_transport
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_fast_stream import verified_fast_plan
from flybrain_drive.flight_perturbation import validate_perturbation_plan
from flybrain_drive.graph import sha256
from flybrain_drive.owned_worker_wait import wait_owned_worker
from scripts.verify_flight_yaw_trial import verified_yaw_trial


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--site-packages', required=True)
    p.add_argument('--block', type=int, required=True, choices=range(3))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--completed-prefix-block',
                   help='Preserve completed independent trials from a terminal failed block')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = read(a.plan)
    validate_perturbation_plan(plan, a.block*4)
    proof = verified_yaw_trial(plan)
    diagnostic_proof = verified_diagnostic_transport(plan)
    fast_proof = verified_fast_plan(plan)
    run = begin_run(a.output, dict(arguments=vars(a), plan=plan, plan_sha256=sha256(a.plan),
                                  yaw_verification=proof, diagnostic_verification=diagnostic_proof,
                                  ui_actions_sent=0, scope=__doc__,
                                  fast_inference_certificate=fast_proof))
    result = dict(status='failed', goal_achieved=False, moving_stabilization_verified=False,
                  ui_actions_sent=0, trials=[])
    if a.completed_prefix_block:
        previous = Path(a.completed_prefix_block)
        previous_result = read(previous / 'result.json')
        previous_manifest = read(previous / 'manifest.json')
        assert previous_result['status'] == 'failed'
        assert previous_manifest['config']['plan'] == plan
        assert previous_manifest['config']['arguments']['block'] == a.block
        completed = previous_result['trials']
        assert [t['index'] for t in completed] == list(range(a.block*4, a.block*4+len(completed)))
        assert len(completed) < 4
        for entry in completed:
            assert sha256(Path(entry['path']) / 'result.json') == entry['result_sha256']
            assert (read(Path(entry['path']) / 'result.json')['status']
                    == 'completed_physical_yaw_trial')
        result['trials'] = list(completed)
        result['preserved_terminal_failed_block'] = dict(path=str(previous),
            result_sha256=sha256(previous / 'result.json'),
            completed_independent_trials=len(completed))

    def stage(name, module, options, gpu=False):
        output = run / name
        bootstrap = ('import sys,site,runpy; p=sys.argv.pop(1); module=sys.argv.pop(1); '
                     'site.addsitedir(p); sys.path.insert(0,p); '
                     'runpy.run_module(module,run_name="__main__")')
        command = ([sys.executable, '-c', bootstrap, a.site_packages, module] if gpu
                   else [sys.executable, '-m', module])
        with (run / (name+'.log')).open('x', encoding='utf8') as log:
            child = subprocess.Popen([*command, *options, '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(run / 'progress.json', dict(stage=name, pid=child.pid,
                                                  completed_trials=result['trials']))
            print(json.dumps(dict(stage=name, pid=child.pid)), flush=True)
            if gpu:
                wait_owned_worker(child, module=module, result_file=output/'result.json',
                                  receipt_file=run/(name+'-process.json'))
            else:
                child.wait()  # Same owned lifecycle child; no timeout replacement.
        receipt = read(output / 'result.json')
        if child.returncode:
            raise RuntimeError('Owned stage failed: '+name+'; '+str(receipt))
        return receipt

    try:
        if not a.execute:
            stage('read-only', 'scripts.probe_flight_moving_body',
                  ['--plan', a.plan, '--perturbation-trial-index', str(a.block*4)], gpu=True)
            result['status'] = 'read_only'
        else:
            for index in range(a.block*4+len(result['trials']), (a.block+1)*4):
                name = f'trial-{index:02d}'
                prepared = stage(name+'-prepare', 'scripts.diagnose_ac_native_ai_pause',
                                 ['--new-episode'])
                assert prepared['status'] == 'observed' and prepared['pause_verified']
                last = read(run / (name+'-prepare') / 'samples.json')[-1]
                assert last['ai']['builtin_ai_controlled'] is False
                receipt = stage(name, 'scripts.probe_flight_moving_body',
                    ['--plan', a.plan, '--perturbation-trial-index', str(index), '--execute'],
                    gpu=True)
                assert receipt['status'] == 'completed_physical_yaw_trial'
                assert receipt['observations'] == 257
                assert receipt['released_neutral_and_pause_verified']
                result['trials'].append(dict(index=index, path=str(run / name),
                    condition=plan['trials'][index],
                    result_sha256=sha256(run / name / 'result.json')))
                write_json(run / 'partial-result.json', result)
            result.update(status='completed_physical_yaw_block', block=a.block)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
