"""Read additional modeled visual columns while proving original complete responses unchanged."""

import argparse
import json
import subprocess
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan['types'] != ['L1', 'L2', 'Mi1', 'Tm3', 'Tm1', 'Tm9']
            or len(plan['episodes']) != 10
            or any(e['role'] not in ('train', 'development') for e in plan['episodes'])):
        raise ValueError('Explicit readouts and unchanged train/development roles required')
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan,
        plan_sha256=sha256(args.plan), program_sha256=sha256(__file__), optimizer_updates=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', episodes=[], optimizer_updates=0, student_deployed=False,
                  goal_achieved=False)
    bootstrap = (
        "import sys,site,runpy; p=sys.argv.pop(1); site.addsitedir(p); sys.path.insert(0,p); "
        "runpy.run_module('scripts.replay_flyvis_cameras',run_name='__main__'); "
        "import gc,torch; torch.cuda.synchronize(); torch.set_default_device('cpu'); "
        "gc.collect(); torch.cuda.empty_cache()")
    try:
        for entry in plan['episodes']:
            for source, digest in plan['source_sha256'].items():
                if sha256(source) != digest:
                    raise ValueError('Configured readout replay sources changed during batch')
            output = run/entry['id']
            command = [plan['python_executable'], '-c', bootstrap, plan['gpu_site'],
                '--recording', entry['recording'], '--body-replay', entry['body_replay'],
                '--model-dir', plan['model_dir'], '--cache-root', plan['cache_root'],
                '--fractional-frozen-tail', '--retinotopic-types', ','.join(plan['types']),
                '--reference-replay', entry['reference_replay'], '--output', str(output)]
            with (run/(entry['id']+'.log')).open('x', encoding='utf8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run/'progress.json', dict(stage=entry['id'], pid=child.pid,
                    completed=len(result['episodes']), total=10, active=True))
                print(json.dumps(dict(stage=entry['id'], pid=child.pid)), flush=True)
                code = child.wait()
            report = read(output/'result.json')
            result['last_child'] = dict(path=str(output), exit_code=code, status=report['status'])
            if (code or report['status'] != 'completed' or report['weights_changed']
                    or report['additional_retinotopic_types'] != plan['types']
                    or max(report['reference_replay_max_errors'].values()) > 2e-6):
                raise ValueError('Complete original responses failed readout-only reproduction')
            result['episodes'].append(dict(entry, output=str(output),
                observations=report['observations'], result_sha256=sha256(output/'result.json'),
                responses_sha256=report['responses_sha256'],
                baseline_errors=report['reference_replay_max_errors']))
            write_json(run/'partial-result.json', result)
        result.update(status='completed_intermediate_visual_readout_batch',
                      author_weights_changed=False, physiological_function_claim=False,
                      original_roles_unchanged=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    finally:
        write_json(run/'progress.json', dict(active=False, status=result['status'],
                                            completed=len(result['episodes']), total=10))
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
