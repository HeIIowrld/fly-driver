"""One fixed future-label probe of the frozen six-muscle steering boundary.

Future teacher commands are training/evaluation targets only. No future sensing,
map, clock index, command or label is added to the actor. This cannot qualify
visual foresight or autonomous driving by itself.
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.constrained_muscle_steering import fit_group_protected_steering
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import MuscleVehicleReadout, read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import metrics


def future_label_join(times, valid, horizon=.2, max_age=.06):
    """Last qualified teacher label at/before each future target, never clamp."""
    t=np.asarray(times,dtype=float); valid=np.asarray(valid)
    if (t.ndim!=1 or len(t)<2 or not np.isfinite(t).all()
            or np.any(np.diff(t)<=0) or valid.shape!=t.shape or valid.dtype!=np.bool_
            or not np.isfinite(horizon) or horizon<=0
            or not np.isfinite(max_age) or max_age<0):
        raise ValueError('Ordered episode times, boolean validity and positive horizon required')
    source=np.flatnonzero(valid)
    if not len(source):raise ValueError('No qualified teacher targets')
    wanted=t+horizon
    lookup=np.searchsorted(t[source],wanted,side='right')-1
    available=(lookup>=0)&(wanted<=t[-1])
    indices=np.full(len(t),-1,dtype=np.int64)
    candidates=source[np.maximum(lookup,0)]
    available &= (t[candidates]>t)&(wanted-t[candidates]<=max_age)
    indices[available]=candidates[available]
    return indices,available


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();plan=read(args.plan);torch.set_num_threads(1)
    if sha256(__file__)!=plan['program_sha256']:raise ValueError('Sealed program changed')
    parent=Path(plan['parent']); result=read(parent/'result.json')
    for name,key in [('native-steering.pt','checkpoint_sha256'),('frozen-trajectories.npz','frozen_trajectories_sha256')]:
        if sha256(parent/name)!=result[key] or sha256(parent/name)!=plan[key]:raise ValueError('Frozen source changed')
    audit=Path(plan['parent_audit'])
    if read(audit/'manifest.json')['config']['fit_result_sha256']!=sha256(parent/'result.json'):
        raise ValueError('Parent independent audit not bound')
    cp=torch.load(parent/'native-steering.pt',map_location='cpu',weights_only=True)
    head=MuscleVehicleReadout(6);head.load_state_dict(cp['readout']);head.eval().requires_grad_(False)
    entries=three_episode_cohort(plan['split']);items=[load_episode(e['data']) for e in entries]
    with np.load(parent/'frozen-trajectories.npz',allow_pickle=False) as z:
        frozen={k:z[k].copy() for k in z.files}
    features=[((torch.tensor(frozen[f'muscles_{e}'])-head.training_reference)/head.training_scale).numpy().astype(float) for e in range(3)]
    joins=[future_label_join(d['decision_time'],d['fit_mask']&d['label_valid'],plan['horizon_seconds'],plan['max_label_age_seconds']) for d,_,_ in items]
    xs=[];ys=[];group_ids=[];group_names=[]
    for e in (0,1):
        d=items[e][0]; ids=np.flatnonzero(d['fit_mask']); offset=sum(len(x) for x in xs)
        xs.append(features[e][ids]);ys.append(d['labels'][ids,0])
        group_ids.extend([np.arange(offset,offset+len(ids)),offset+np.flatnonzero(frozen[f'pre_{e}'][ids])])
        group_names.extend([f'train{e}_present_all',f'train{e}_present_pre'])
        j,ok=joins[e];ids=np.flatnonzero(frozen[f'pre_{e}']&ok)
        if not len(ids):raise ValueError('No qualified future pre-corner targets')
        offset=sum(len(x) for x in xs);xs.append(features[e][ids]);ys.append(d['labels'][j[ids],0])
        group_ids.append(np.arange(offset,offset+len(ids)));group_names.append(f'train{e}_future_pre')
    x=np.concatenate(xs);y=np.concatenate(ys);groups=[]
    for ids in group_ids:
        mask=np.zeros(len(x),dtype=bool);mask[ids]=True;groups.append(mask)
    run=begin_run(args.output,dict(plan=plan,plan_sha256=sha256(args.plan),entries=entries,
        seed=None,training_episodes=[0,1],development_episode=2,group_names=group_names,
        scope='Consumed Monza; frozen representation already trained on these corners. Not new-corner or live qualification.',
        actor_inputs_changed=False,vehicle_control_sent=False,actor_qualified=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    shutil.copyfile('src/flybrain_drive/constrained_muscle_steering.py',run/'executed-fit.py')
    old={k:v.clone() for k,v in head.state_dict().items()}
    output=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
    try:
        fit=fit_group_protected_steering(x,y,groups,old['projection.weight'][0].numpy(),float(old['projection.bias'][0]))
        write_json(run/'train-fit.json',{k:v for k,v in fit.items() if k not in ('beta','mean','scale')})
        with torch.no_grad():
            head.projection.weight[0].copy_(torch.tensor(fit['beta'],dtype=torch.float32))
            head.projection.bias[0]=fit['bias']
        for k in ('projection.weight','projection.bias'):assert torch.equal(old[k][1:],head.state_dict()[k][1:])
        for k in ('training_reference','training_scale'):assert torch.equal(old[k],head.state_dict()[k])
        reports=[];arrays={}
        for e,(d,_,_) in enumerate(items):
            mu=torch.tensor(frozen[f'muscles_{e}'])
            original=MuscleVehicleReadout(6);original.load_state_dict(old)
            with torch.no_grad():baseline=original(mu).numpy();candidate=head(mu).numpy()
            assert np.array_equal(baseline[:,1:],candidate[:,1:])
            j,ok=joins[e];mask=frozen[f'pre_{e}']&ok
            current=lambda p:metrics(p[:,0],d,frozen[f'pre_{e}'])
            future=lambda p:float(np.sqrt(np.mean((p[mask,0]-d['labels'][j[mask],0])**2)))
            reports.append(dict(episode=e,baseline=current(baseline),candidate=current(candidate),
                future_pre_baseline_rmse=future(baseline),future_pre_candidate_rmse=future(candidate),
                future_pre_count=int(mask.sum()),pedals_exact=True))
            arrays.update({f'baseline_{e}':baseline,f'candidate_{e}':candidate,f'future_indices_{e}':j,f'future_valid_{e}':ok})
        np.savez_compressed(run/'predictions.npz',**arrays)
        child=dict(cp);child['readout']=head.state_dict();child['method']='protected_present_and_future_pre_200ms'
        child['actor_qualified']=False;torch.save(child,run/'native-steering.pt')
        dev=reports[2]
        passed=(all(dev['candidate'][k]<=.98*dev['baseline'][k] for k in ('all_rmse','pre_rmse'))
            and dev['future_pre_candidate_rmse']<=.98*dev['future_pre_baseline_rmse'])
        output.update(status='completed_protected_anticipatory_probe',comparisons=reports,
            group_names=group_names,group_counts=[int(g.sum()) for g in groups],
            minimum_training_mse_improvement=fit['minimum_training_mse_improvement'],
            maximum_coefficient_change=float(np.max(abs(head.projection.weight[0].detach().numpy()-old['projection.weight'][0].numpy()))),
            development_present_and_future_two_percent_gate_passed=passed,
            checkpoint_sha256=sha256(run/'native-steering.pt'),predictions_sha256=sha256(run/'predictions.npz'),
            termination_reason='One fixed TRAIN-only constrained fit; consumed development evaluated once; no deployment')
    except Exception as error:output.update(error=type(error).__name__,reason=str(error),termination_reason='Explicit failure; no fallback fit')
    write_json(run/'result.json',output)
    if output['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
