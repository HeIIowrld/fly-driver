"""Every original full history, five fast-path hypotheses and paired yaw interventions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.owned_worker_wait import wait_owned_worker


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--refinement-audit', required=True)
    parser.add_argument('--site-packages', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.diagnosis)
    diagnosis = read(source/'result.json')
    numerical = read(Path(args.refinement_audit)/'result.json')
    assert diagnosis['status'] == 'completed_original_history_decomposition'
    assert numerical['all_cases_passed']
    assert len(diagnosis['trials']) == 6
    plan_sha, ref_sha = sha256(args.plan), sha256(args.reference)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        source_result_sha256=sha256(source/'result.json'), plan_sha256=plan_sha,
        reference_sha256=ref_sha, numerical_audit_sha256=sha256(
            Path(args.refinement_audit)/'result.json'),
        selection='Every full-mode original: both signs and all3 blocks; no outcome selection'))
    result = dict(status='failed', goal_achieved=False, actor_qualified=False,
        actuator_messages_sent=0, optimizer_updates=0, ui_actions_sent=0, trials=[])
    try:
        for entry in diagnosis['trials']:
            assert sha256(args.plan) == plan_sha and sha256(args.reference) == ref_sha
            old = Path(entry['path'])
            assert sha256(old/'result.json') == entry['result_sha256']
            old_result = read(old/'result.json')
            assert (sha256(old/'counterfactuals.npz')
                    == old_result['counterfactual_diagnosis']['archive_sha256'])
            old_config = read(old/'manifest.json')['config']
            recording = old_config['arguments']['recording']
            index = old_result['trial_index']
            output = run/f'trial-{index:02d}'
            bootstrap = ('import sys,site,runpy; p=sys.argv.pop(1); site.addsitedir(p); '
                'sys.path.insert(0,p); '
                'runpy.run_module("scripts.audit_flight_physical_yaw_trial",run_name="__main__")')
            with (run/f'trial-{index:02d}.log').open('x', encoding='utf8') as log:
                child = subprocess.Popen([sys.executable, '-c', bootstrap, args.site_packages,
                    '--recording', recording, '--fast-plan', args.plan,
                    '--fast-reference', args.reference, '--output', str(output)],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run/'progress.json', dict(index=index, pid=child.pid,
                                                    completed_trials=result['trials']))
                print(json.dumps(dict(index=index, pid=child.pid)), flush=True)
                code = wait_owned_worker(child, module='scripts.audit_flight_physical_yaw_trial',
                    result_file=output/'result.json',
                    receipt_file=run/f'trial-{index:02d}-process.json')
            assert code == 0
            receipt = read(output/'result.json')
            proof = receipt['fast_original_replay']
            assert sha256(output/'fast-hypotheses.npz') == proof['archive_sha256']
            with (load_array(old/'counterfactuals.npz', allow_pickle=False) as before,
                  load_array(output/'fast-hypotheses.npz', allow_pickle=False) as after):
                np.testing.assert_array_equal(before['original_clock'], after['original_clock'])
                np.testing.assert_allclose(after['intents'][:, 0, 0], before['intents'][:, 0],
                                           atol=3e-6, rtol=2e-5)
                np.testing.assert_allclose(after['intents'][:, 0, 1], before['intents'][:, 1],
                                           atol=3e-6, rtol=2e-5)
                zero_error = float(np.max(np.abs(
                    after['intents'][:, 0, 1]-before['intents'][:, 1])))
                moving = np.isin(after['phase_index'], [1, 2, 3, 4])
                maximum = np.max(np.abs(after['intents'][moving, :, 0, 0]), axis=0).tolist()
            result['trials'].append(dict(index=index, path=str(output),
                result_sha256=sha256(output/'result.json'),
                original_neuron_max_error=proof['maximum_original_neuron_error'],
                original_full_intent_max_error=proof['maximum_original_intent_error'],
                old_zero_yaw_intent_max_error=zero_error,
                known_moving_history_max_abs_left_intent_by_hypothesis=maximum))
            write_json(run/'partial-result.json', result)
        result.update(status='completed_fast_original_history_audits', original_histories=6,
            original_observations=1542, hypothetical_outputs=15420,
            original_clocks_pixels_native_preserved=True, physical_trajectories_created=0,
            physical_stability_repaired=False, physiological_equivalence_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
