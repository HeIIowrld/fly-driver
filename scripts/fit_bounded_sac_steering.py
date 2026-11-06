"""TRAIN-only selection of command-space six-muscle steering readouts.

Cross-lap selection is conditional on an already trained representation, NOT an
unseen-corner evaluation. The consumed third lap is evaluated only after choice.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.bounded_steering_fit import fit_bounded_steering
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import MuscleVehicleReadout, read
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment, sustained_onsets
from scripts.sac_steering_experiment import dataset

PARENT = Path('runs/steering-precorners-20261103/optical-pre16-fit-01')
PARENT_SHA = 'bb0bb4dd378e1685594e07928e3cd92ac8bf2543da6665c81aa5f05980fb6384'


def pre_mask(d, alignment):
    mask = np.zeros(len(d['labels']), dtype=bool)
    for onset in sustained_onsets(d, alignment):
        mask |= ((d['decision_time'] >= onset['time']-.8)
                 & (d['decision_time'] <= onset['time']-.08)
                 & (abs(d['body'][:, 4]) < .07))
    return mask & d['fit_mask']


def metrics(prediction, data, pre):
    if not pre.any():
        raise ValueError('No sustained pre-turn supervision; do not infer a zero error')
    error = prediction - data['labels'][:, 0]
    return dict(all_rmse=float(np.sqrt(np.mean(error[data['fit_mask']]**2))),
                pre_rmse=float(np.sqrt(np.mean(error[pre]**2))),
                all_count=int(data['fit_mask'].sum()), pre_count=int(pre.sum()))


def ridge(x, y, weights):
    mean = np.average(x, axis=0, weights=weights)
    ym = np.average(y, weights=weights)
    scale = x.std(0); scale = np.where(scale > 1e-8, scale, 1.)
    z = (x-mean)/scale
    beta = np.linalg.solve(z.T @ (weights[:, None]*z) + .001*weights.sum()*np.eye(6),
                           z.T @ (weights*(y-ym))) / scale
    return beta, float(ym-mean @ beta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(780510)
    if sha256(PARENT/'steering.pt') != PARENT_SHA:
        raise ValueError('Protected parent changed')
    receipt = read(PARENT/'result.json')
    if sha256(PARENT/'predictions.npz') != receipt['predictions_sha256']:
        raise ValueError('Parent fixed neural trajectories changed')
    entries, items, _ = dataset()
    alignment, sign_receipt = steering_yaw_alignment([x[0] for x in items[:2]])
    cp = torch.load(PARENT/'steering.pt', map_location='cpu', weights_only=True)
    head = MuscleVehicleReadout(6); head.load_state_dict(cp['readout'])
    with np.load(PARENT/'predictions.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    keys = ['train_0', 'train_1', 'dev']
    features = [((torch.tensor(arrays[k+'_muscles'])-head.training_reference)/head.training_scale).numpy().astype(float) for k in keys]
    pre = [pre_mask(d, alignment) for d, _, _ in items]
    run = begin_run(a.output, dict(seed=780510, program_sha256=sha256(__file__),
        parent_checkpoint_sha256=PARENT_SHA, entries=entries, sign_evidence=sign_receipt,
        selection='Two conditional readout cross-lap folds within TRAIN only; representation previously trained on both. No unseen-corner claim.',
        objective='Weighted actual tanh steering MSE plus normalized coefficient ridge; six muscle summaries only',
        grid=dict(preweights=[4,16,64], regularization=[.0001,.001]),
        selection_score='max(mean fold whole-RMSE ratio, mean fold pre-RMSE ratio) against fold-local atanh ridge pre16 .001',
        gate='Final consumed development whole and sustained-pre RMSE both >=2% lower than parent; pedals exact; not live qualification',
        actor_deployed=False, vehicle_control_sent=False))
    shutil.copyfile(__file__, run/'executed-entrypoint.py')
    shutil.copyfile('src/flybrain_drive/bounded_steering_fit.py', run/'executed-fit.py')
    start = time.monotonic(); result = dict(status='failed', actor_qualified=False, vehicle_control_sent=False)
    try:
        baseline_folds = []; folds = []
        for fit_i, test_i in [(0,1), (1,0)]:
            d = items[fit_i][0]; m = d['fit_mask']
            b, bias = ridge(features[fit_i][m], np.arctanh(d['labels'][m,0]), np.where(pre[fit_i][m],16.,1.))
            baseline_folds.append(metrics(np.tanh(features[test_i]@b+bias), items[test_i][0], pre[test_i]))
            folds.append((fit_i, test_i, b, bias))
        candidates = []
        for weight in [4,16,64]:
            for penalty in [.0001,.001]:
                reports = []
                for fold_i, (fit_i,test_i,b,bias) in enumerate(folds):
                    d = items[fit_i][0]; m = d['fit_mask']
                    fit = fit_bounded_steering(features[fit_i][m], d['labels'][m,0],
                        np.where(pre[fit_i][m],float(weight),1.), b, bias, penalty)
                    report = metrics(np.tanh(features[test_i]@fit['beta']+fit['bias']),items[test_i][0],pre[test_i])
                    reports.append(report)
                ratios = [float(np.mean([r[k]/b[k] for r,b in zip(reports,baseline_folds)])) for k in ['all_rmse','pre_rmse']]
                candidates.append(dict(preweight=weight, regularization=penalty, folds=reports,
                                       mean_ratios=ratios, score=float(max(ratios))))
        chosen = min(candidates, key=lambda c:c['score'])
        # This receipt precedes ANY third-lap prediction/label metric calculation.
        write_json(run/'train-selection.json', dict(candidates=candidates,chosen=chosen,baseline_folds=baseline_folds,
                    development_used_for_selection=False,representation_seen_both_training_laps=True))
        masks = [items[i][0]['fit_mask'] for i in [0,1]]
        x = np.concatenate([features[i][masks[i]] for i in [0,1]])
        y = np.concatenate([items[i][0]['labels'][masks[i],0] for i in [0,1]])
        weights = np.concatenate([np.where(pre[i][masks[i]],float(chosen['preweight']),1.) for i in [0,1]])
        initial_beta = head.projection.weight[0].detach().numpy().copy()
        initial_bias = float(head.projection.bias[0].detach())
        fit = fit_bounded_steering(x,y,weights,initial_beta,initial_bias,chosen['regularization'])
        old = {k:v.clone() for k,v in head.state_dict().items()}
        with torch.no_grad():
            head.projection.weight[0] = torch.tensor(fit['beta'],dtype=torch.float32)
            head.projection.bias[0] = fit['bias']
        comparisons = []
        for i,key in enumerate(keys):
            old_prediction = arrays[key].copy()
            new_prediction = head(torch.tensor(arrays[key+'_muscles'])).detach().numpy()
            np.testing.assert_allclose(new_prediction[:,1:],old_prediction[:,1:],atol=2e-7,rtol=0)
            # Keep saved pedal predictions byte-exact, avoiding batching roundoff.
            arrays[key][:,0] = new_prediction[:,0]
            comparisons.append(dict(parent=metrics(old_prediction[:,0],items[i][0],pre[i]),
                                    candidate=metrics(new_prediction[:,0],items[i][0],pre[i])))
        for k in ['projection.weight','projection.bias']:
            assert torch.equal(old[k][1:],head.state_dict()[k][1:])
        for k in ['training_reference','training_scale']:
            assert torch.equal(old[k],head.state_dict()[k])
        cp.update(readout=head.state_dict(),parent_checkpoint_sha256=PARENT_SHA,
                  actor_qualified=False,head_objective='bounded_command_mse',head_selection=chosen)
        torch.save(cp,run/'steering.pt')
        torch.load(run/'steering.pt',map_location='cpu',weights_only=True)
        np.savez_compressed(run/'predictions.npz',**arrays)
        dev = comparisons[2]
        passed = all(dev['candidate'][k] <= .98*dev['parent'][k] for k in ['all_rmse','pre_rmse'])
        result.update(status='completed_bounded_steering_readout',comparisons=comparisons,
            chosen=chosen,development_two_percent_gate_passed=passed,pedal_rows_exact=True,
            neural_states_reused_exact=True,readout_only_crossfit_not_unseen_corner=True,
            checkpoint_sha256=sha256(run/'steering.pt'),predictions_sha256=sha256(run/'predictions.npz'),
            objective_evaluations=fit['objective_evaluations'],fit_initial_loss=fit['initial_loss'],fit_final_loss=fit['final_loss'],
            termination_reason='TRAIN-only readout choice sealed, full train and consumed dev evaluated; no live/physiology/generalization qualification')
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit fit or invariant failure')
    result['elapsed_seconds'] = time.monotonic()-start
    write_json(run/'result.json',result); print(json.dumps(result))
    if result['status']=='failed': raise SystemExit(1)


if __name__ == '__main__': main()
