"""Diagnostic body-only predictors; never substitute for the flight actor."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode
from scripts.probe_protected_anticipatory_steering import future_label_join


def body_features(d,history):
    b=np.where(d['body_valid'],d['body'],0.)
    mask=d['body_valid'].astype(float)
    if not history:return np.column_stack([b,mask])
    past=np.where(d['view_body_valid'],d['view_body'],0.).reshape(len(b),-1)
    valid=d['view_body_valid'].reshape(len(b),-1).astype(float)
    return np.column_stack([b,mask,past,valid])


def fit_predict(x,y,others):
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    z=(x-mean)/scale;center=y.mean()
    beta=np.linalg.solve(z.T@z+.01*len(z)*np.eye(z.shape[1]),z.T@(y-center))
    return [np.clip(((v-mean)/scale)@beta+center,-1,1) for v in others],dict(mean=mean,scale=scale,beta=beta,bias=np.array(center))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    probe=Path(a.probe);audit=read(probe.parent/'audit-01/result.json')
    assert audit['status']=='completed_independent_anticipatory_probe_audit'
    assert read(probe.parent/'audit-01/manifest.json')['config']['fit_result_sha256']==sha256(probe/'result.json')
    cfg=read(probe/'manifest.json')['config'];plan=cfg['plan']
    entries=three_episode_cohort(plan['split']);assert entries==cfg['entries']
    items=[load_episode(e['data']) for e in entries]
    joins=[future_label_join(d['decision_time'],d['fit_mask']&d['label_valid'],.2,.06) for d,_,_ in items]
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),probe_result_sha256=sha256(probe/'result.json'),
        entries=entries,training_episodes=[0,1],development_episode=2,seed=None,
        regularization=.01,selection='One fixed ridge per input/target definition; no development tuning',
        scope='Diagnostic information availability, not attribution to the student, new-corner proof or biological actor',
        actor_replaced=False,vehicle_control_sent=False,optimizer_updates=0))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    reports=[];saved={}
    with np.load(Path(plan['parent'])/'frozen-trajectories.npz',allow_pickle=False) as frozen:
        for history in (False,True):
            features=[body_features(d,history) for d,_,_ in items]
            for future in (False,True):
                masks=[d['fit_mask']&(joins[e][1] if future else True) for e,(d,_,_) in enumerate(items)]
                ys=[d['labels'][np.maximum(joins[e][0],0),0] if future else d['labels'][:,0] for e,(d,_,_) in enumerate(items)]
                x=np.concatenate([features[e][masks[e]] for e in (0,1)])
                y=np.concatenate([ys[e][masks[e]] for e in (0,1)])
                predictions,state=fit_predict(x,y,features);name=('history' if history else 'current')+('_future200' if future else '_present')
                for k,v in state.items():saved[name+'_'+k]=v
                values=[]
                constant=float(y.mean())
                for e,(d,_,_) in enumerate(items):
                    whole=masks[e];pre=frozen[f'pre_{e}']&whole;pred=predictions[e]
                    values.append(dict(episode=e,all_rmse=float(np.sqrt(np.mean((pred[whole]-ys[e][whole])**2))),
                        pre_rmse=float(np.sqrt(np.mean((pred[pre]-ys[e][pre])**2))),
                        constant_all_rmse=float(np.sqrt(np.mean((constant-ys[e][whole])**2))),
                        constant_pre_rmse=float(np.sqrt(np.mean((constant-ys[e][pre])**2))),
                        all_count=int(whole.sum()),pre_count=int(pre.sum())))
                    saved[name+f'_prediction_{e}']=pred
                reports.append(dict(name=name,input_dimensions=features[0].shape[1],comparisons=values))
    np.savez_compressed(run/'predictors.npz',**saved)
    write_json(run/'result.json',dict(status='completed_body_only_forecast_diagnostic',reports=reports,
        predictors_sha256=sha256(run/'predictors.npz'),actor_replaced=False,vehicle_control_sent=False,
        termination_reason='Fixed TRAIN-only body-only predictors evaluated on consumed development; no model or simulator changes'))


if __name__=='__main__':main()
