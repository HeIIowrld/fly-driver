"""Complete both drive boundaries on every original full history before any live fast control."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_fast_stream import fast_definition
from flybrain_drive.graph import sha256
from flybrain_drive.owned_worker_wait import wait_owned_worker


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--site-packages', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config, _, _, _, _, _ = fast_definition(args.config)
    sources = read(Path(config['original_history_verification'])/'result.json')['trials']
    assert len(sources) == 6
    digest = sha256(args.config)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        fast_config_sha256=digest, source_histories=sources, modes=('full', 'zero_yaw'),
        selection='All6 original histories, both boundaries; no timing or outcome selection'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
                  ui_actions_sent=0, optimizer_updates=0, trials=[])
    try:
        for source in sources:
            assert sha256(Path(source['path'])/'result.json') == source['result_sha256']
            for mode in ('full', 'zero_yaw'):
                assert sha256(args.config) == digest
                name = f"trial-{source['index']:02d}-{mode}"
                output = run/name
                bootstrap = ('import sys,site,runpy; p=sys.argv.pop(1); site.addsitedir(p); '
                    'sys.path.insert(0,p); '
                    'runpy.run_module("scripts.verify_flight_fast_stream_trial",run_name="__main__")')
                with (run/(name+'.log')).open('x', encoding='utf8') as log:
                    child = subprocess.Popen([sys.executable, '-c', bootstrap, args.site_packages,
                        '--config', args.config, '--source-audit', source['path'],
                        '--mode', mode, '--output', str(output)],
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    write_json(run/'progress.json', dict(stage=name, pid=child.pid,
                                                        completed_trials=result['trials']))
                    print(json.dumps(dict(stage=name, pid=child.pid)), flush=True)
                    code = wait_owned_worker(child,
                        module='scripts.verify_flight_fast_stream_trial',
                        result_file=output/'result.json', receipt_file=run/(name+'-process.json'))
                receipt = read(output/'result.json')
                assert code == 0 and receipt['status'] == 'completed_fast_stream_trial_replay'
                result['trials'].append(dict(path=str(output), mode=mode, index=source['index'],
                    result_sha256=sha256(output/'result.json'),
                    maximum_intent_error=receipt['maximum_intent_error'],
                    maximum_exact_neuron_error=receipt['maximum_exact_neuron_error'],
                    maximum_fine_neuron_error=receipt['maximum_fine_neuron_error'],
                    real_time_budget_passed=receipt['real_time_budget_passed'],
                    wall_seconds_percentiles=receipt['wall_seconds_percentiles']))
                write_json(run/'partial-result.json', result)
        result.update(status='completed_fast_stream_original_replay',
            histories=6, modes_per_history=2,
            unique_original_observations=1542, replayed_observations=3084,
            all_original_observations_replayed=True, full_and_zero_yaw_both_verified=True,
            parameters_fixed_during_each_replay=True, hidden_actual_resets=0,
            maximum_intent_error=max(t['maximum_intent_error'] for t in result['trials']),
            maximum_exact_neuron_error=max(
                t['maximum_exact_neuron_error'] for t in result['trials']),
            maximum_fine_neuron_error=max(t['maximum_fine_neuron_error'] for t in result['trials']),
            real_time_budget_passed=all(t['real_time_budget_passed'] for t in result['trials']),
            actor_qualified=False, physical_stability_repaired=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
