"""Single TRAIN whole/pre group-constrained fit of frozen native flight muscles."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.constrained_muscle_steering import fit_group_protected_steering
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import MuscleVehicleReadout,read
from scripts.fit_bounded_sac_steering import metrics
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1)
    assert sha256(__file__)==plan['program_sha256']
    previous=Path(plan['previous_fit']);pr=read(previous/'result.json')
    assert pr['status']=='completed_native_flight_steering_refit'
    assert sha256(previous/'frozen-trajectories.npz')==pr['frozen_trajectories_sha256']==plan['frozen_trajectories_sha256']
    audit=Path(plan['previous_audit']);assert read(audit/'result.json')['status']=='completed_independent_native_flight_steering_refit_audit'
    assert read(audit/'manifest.json')['config']['fit_result_sha256']==sha256(previous/'result.json')
    native=Path(plan['native_fit']);nr=read(native/'result.json');nc=read(native/'manifest.json')['config']
    assert sha256(native/'body-extension.pt')==nr['checkpoint_sha256']==plan['native_checkpoint_sha256']
    parent=torch.load(native/'body-extension.pt',map_location='cpu',weights_only=True)
    source=Path(parent['source_fit_directory']);assert sha256(source/'steering.pt')==parent['source_checkpoint_sha256']
    cp=torch.load(source/'steering.pt',map_location='cpu',weights_only=True)
    head=MuscleVehicleReadout(6);head.load_state_dict(cp['readout']);head.eval().requires_grad_(False)
    entries=three_episode_cohort(nc['arguments']['split']);items=[load_episode(e['data']) for e in entries]
    with np.load(previous/'frozen-trajectories.npz',allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
    features=[((torch.tensor(saved[f'muscles_{e}'])-head.training_reference)/head.training_scale).numpy().astype(float) for e in range(3)]
    train_x=[];train_y=[];sizes=[]
    for e in (0,1):
        d=items[e][0];mask=d['fit_mask'];train_x.append(features[e][mask]);train_y.append(d['labels'][mask,0]);sizes.append(int(mask.sum()))
    x=np.concatenate(train_x);y=np.concatenate(train_y);groups=[]
    for e in (0,1):
        mask=np.zeros(len(x),dtype=bool);start=0 if e==0 else sizes[0];mask[start:start+sizes[e]]=True
        groups.append(mask);pre=mask.copy();pre[mask]=saved[f'pre_{e}'][items[e][0]['fit_mask']];groups.append(pre)
    run=begin_run(a.output,dict(plan=plan,plan_sha256=sha256(a.plan),entries=entries,seed=None,
        program_sha256=sha256(__file__),method='train_group_constraints',protected_groups=['train0_all','train0_pre','train1_all','train1_pre'],
        selection='One max-min normalized TRAIN MSE improvement fit; no development labels in fit or configuration choice',
        actor_qualified=False,vehicle_control_sent=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');shutil.copyfile('src/flybrain_drive/constrained_muscle_steering.py',run/'executed-fit.py')
    shutil.copyfile(previous/'frozen-trajectories.npz',run/'frozen-trajectories.npz')
    out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,method='train_group_constraints')
    try:
        old={k:v.clone() for k,v in head.state_dict().items()}
        fitted=fit_group_protected_steering(x,y,groups,old['projection.weight'][0].numpy(),float(old['projection.bias'][0]))
        write_json(run/'train-fit.json',{k:v for k,v in fitted.items() if k not in ('beta','mean','scale')})
        with torch.no_grad():
            head.projection.weight[0].copy_(torch.tensor(fitted['beta'],dtype=torch.float32));head.projection.bias[0]=fitted['bias']
        for k in ('projection.weight','projection.bias'):assert torch.equal(old[k][1:],head.state_dict()[k][1:])
        for k in ('training_reference','training_scale'):assert torch.equal(old[k],head.state_dict()[k])
        comparisons=[]
        for e,(d,c,_) in enumerate(items):
            raw=head(torch.tensor(saved[f'muscles_{e}'])).detach().numpy();baseline=saved[f'baseline_{e}']
            np.testing.assert_allclose(raw[:,1:],baseline[:,1:],rtol=0,atol=2e-7)
            values=baseline.copy();values[:,0]=raw[:,0];saved[f'candidate_{e}']=values
            comparisons.append(dict(episode=e,baseline=metrics(baseline[:,0],d,saved[f'pre_{e}']),candidate=metrics(values[:,0],d,saved[f'pre_{e}']),pedals_exact=bool(np.array_equal(values[:,1:],baseline[:,1:]))))
        for r in comparisons[:2]:
            assert all(r['candidate'][k]<=r['baseline'][k]*(1+1e-6) for k in ('all_rmse','pre_rmse'))
        torch.save(dict(schema='flybrain.native_flight_steering_refit.v1',native_checkpoint_sha256=plan['native_checkpoint_sha256'],native_fit_directory=str(native),
            readout=head.state_dict(),body_port=parent['body_port'],source_checkpoint_sha256=parent['source_checkpoint_sha256'],actor_qualified=False,head_fit_train_only=True,method='train_group_constraints'),run/'native-steering.pt')
        torch.load(run/'native-steering.pt',map_location='cpu',weights_only=True)
        np.savez_compressed(run/'predictions.npz',**saved)
        dev=comparisons[2];gate=all(dev['candidate'][k]<=.98*dev['baseline'][k] for k in ('all_rmse','pre_rmse'))
        out.update(status='completed_native_flight_steering_refit',comparisons=comparisons,development_two_percent_gate_passed=gate,
            minimum_training_mse_improvement=fitted['minimum_training_mse_improvement'],group_mse_ratios=fitted['group_mse_ratios'],
            objective_evaluations=fitted['objective_evaluations'],checkpoint_sha256=sha256(run/'native-steering.pt'),
            predictions_sha256=sha256(run/'predictions.npz'),frozen_trajectories_sha256=sha256(run/'frozen-trajectories.npz'),
            termination_reason='One TRAIN group-protected fit on unchanged full native flight trajectories; consumed development evaluated after fit, not live/unseen-track qualification')
    except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit protected fit/invariant failure; no fallback or deployment')
    write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
