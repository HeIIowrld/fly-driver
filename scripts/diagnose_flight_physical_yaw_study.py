"""Exploratory full-history decomposition of every full trial; no actuator or model updates."""

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

LAGS = (0., .04, .08, .12, .16, .2, .28, .4, .6)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def correlation(first, second):
    if len(first) < 5 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def decompose(path):
    result = read(path/'result.json')
    proof = result['counterfactual_diagnosis']
    assert sha256(path/'counterfactuals.npz') == proof['archive_sha256']
    with load_array(path/'counterfactuals.npz', allow_pickle=False) as data:
        times, yaw, intents = data['original_clock'], data['yaw'], data['intents']
        assert data['modes'].tolist() == ['full', 'zero_yaw', 'reversed_yaw',
            'scene_command_lesion', 'zero_yaw_scene_command_lesion']
        # The flight circuit emits turn and forward intent; pedals are an adapter boundary.
        assert intents.shape == (257, 5, 2) and np.all(np.diff(times) > 0)
    on, off = result['pulse_on_host_clock'], result['pulse_off_host_clock']
    baseline = (times >= on-.3) & (times < on)
    selected = (times >= on) & (times <= off+1.6)
    assert baseline.sum() >= 5
    body = intents[:, 0, 0]-intents[:, 1, 0]
    scene = intents[:, 0, 0]-intents[:, 3, 0]
    interaction = body-(intents[:, 3, 0]-intents[:, 4, 0])
    yaw_relative = yaw-yaw[baseline].mean()
    body_relative = body-body[baseline].mean()
    qualified = selected & (np.abs(yaw_relative) >= .005) & (np.abs(body_relative) >= 1e-6)
    lags = []
    for lag in LAGS:
        indices = np.searchsorted(times, times-lag, side='right')-1
        included = selected & (indices >= 0)
        current = np.flatnonzero(included)
        previous = indices[included]
        actual = times[current]-times[previous]
        lags.append(dict(requested_lag_seconds=lag, pairs=len(current),
            actual_lag_seconds_percentiles=np.percentile(actual, [0, 50, 100]).tolist(),
            opposition_correlation=correlation(-body_relative[current], yaw_relative[previous])))
    return dict(trial=result['trial'], trial_index=result['trial_index'],
        full_duplicate_max_error=proof['full_batch_maximum_error'],
        pulse_body_intent_rms=float(np.sqrt(np.mean(body_relative[selected]**2))),
        pulse_scene_intent_rms=float(np.sqrt(np.mean((scene-scene[baseline].mean())[selected]**2))),
        baseline_body_intent_mean=float(body[baseline].mean()),
        baseline_scene_intent_mean=float(scene[baseline].mean()),
        maximum_body_steering_change_fraction_of_pulse=float(
            .6*np.max(np.abs(body_relative[selected]))/.05),
        baseline_subtracted_interaction_rms=float(np.sqrt(np.mean(
            (interaction-interaction[baseline].mean())[selected]**2))),
        excited_observations=int(qualified.sum()),
        contemporaneous_opposition_fraction=(float(np.mean(
            body_relative[qualified]*yaw_relative[qualified] < 0)) if qualified.any() else None),
        lagged_correlations=lags,
        physical_counterfactual_trajectory=False, physiology_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--site-packages', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--completed-trial0',
                        help='Reuse an actually terminated completed trial0 audit')
    args = parser.parse_args()
    study = read(Path(args.study)/'result.json')
    assert study['full_study_complete'] and len(study['blocks']) == 3
    recordings = []
    for entry in study['blocks']:
        block = read(Path(entry['original'])/'result.json')
        audit = read(Path(entry['audit'])/'result.json')
        assert sha256(Path(entry['audit'])/'result.json') == entry['audit_sha256']
        for trial in block['trials']:
            if trial['condition']['mode'] == 'full':
                assert sha256(Path(trial['path'])/'result.json') == trial['result_sha256']
                audited = next(t for t in audit['audits'] if t['index'] == trial['index'])
                assert sha256(Path(audited['path'])/'result.json') == audited['result_sha256']
                recordings.append(trial)
    assert len(recordings) == 6
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        selection='All six full-mode trials, both signs and every block; no outcome selection',
        source_sha256=sha256(Path(args.study)/'result.json'), recordings=recordings,
        lags_seconds=LAGS,
        lag_sampling='Previous actual original-clock observation, no interpolation',
        analysis_scope='Post-study exploratory diagnosis, never a replacement physical gate'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
        ui_actions_sent=0, optimizer_updates=0,
        source_physical_gate_passed=study['physical_yaw_disturbance_iae_gate_passed'], trials=[])
    try:
        if args.completed_trial0:
            previous = Path(args.completed_trial0)
            receipt = read(previous/'result.json')
            arguments = read(previous/'manifest.json')['config']['arguments']
            process = read(previous.parent/'trial-00-process.json')
            assert receipt['status'] == 'completed_physical_yaw_trial_audit'
            assert receipt['actuator_messages_sent'] == 0 and receipt['trial_index'] == 0
            assert arguments['counterfactuals'] and arguments['recording'] == recordings[0]['path']
            assert process['process']['termination_signaled'] and process['child_returncode'] == 0
            result['trials'].append(dict(path=str(previous), analysis=decompose(previous),
                result_sha256=sha256(previous/'result.json'),
                preserved_failed_aggregation=read(previous.parent/'result.json')['status']))
            write_json(run/'partial-result.json', result)
        for entry in recordings[len(result['trials']):]:
            output = run/f"trial-{entry['index']:02d}"
            bootstrap = ('import sys,site,runpy; p=sys.argv.pop(1); site.addsitedir(p); '
                'sys.path.insert(0,p); '
                'runpy.run_module("scripts.audit_flight_physical_yaw_trial",run_name="__main__")')
            with (run/f"trial-{entry['index']:02d}.log").open('x', encoding='utf8') as log:
                child = subprocess.Popen([sys.executable, '-c', bootstrap, args.site_packages,
                    '--recording', entry['path'], '--counterfactuals', '--output', str(output)],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run/'progress.json', dict(index=entry['index'], pid=child.pid,
                                                    completed_trials=result['trials']))
                print(json.dumps(dict(index=entry['index'], pid=child.pid)), flush=True)
                code = wait_owned_worker(child, module='scripts.audit_flight_physical_yaw_trial',
                    result_file=output/'result.json',
                    receipt_file=run/f"trial-{entry['index']:02d}-process.json")
            assert code == 0
            result['trials'].append(dict(path=str(output), analysis=decompose(output),
                                         result_sha256=sha256(output/'result.json')))
            write_json(run/'partial-result.json', result)
        result.update(status='completed_original_history_decomposition', trials_completed=6,
            actual_physical_stability_repaired=False, navigation_verified=False,
            actor_qualified=False, original_histories_preserved=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
