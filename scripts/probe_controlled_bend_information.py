"""TRAIN-only linear comparison probes, never an actor or measured circuit."""
import argparse,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.fine_motion_history import FineMotionHistory
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def fit_probe(x,y,ridge,floor):
    reference=x.mean(0);scale=np.maximum(x.std(0),floor);z=np.clip((x-reference)/scale,-5,5);center=z.mean(0);a=z-center;target=y-y.mean();weight=a.T@np.linalg.solve(a@a.T+ridge*np.eye(len(a)),target);bias=float(y.mean()-center@weight)
    return dict(reference=reference,scale=scale,weight=weight,bias=np.array(bias))


def scores(x,fit):return np.clip((x-fit['reference'])/fit['scale'],-5,5)@fit['weight']+fit['bias']


def report(value,bends,late,prefix):
    means=value[:,late].mean(1);pairs=[]
    for i in range(0,len(means),2):
        assert bends[i]==-1 and bends[i+1]==1;pairs.append(float((means[i+1]-means[i])/2))
    return dict(positive_bend_pairs=sum(x>1e-6 for x in pairs),pairs=len(pairs),signed_pair_margins=pairs,absolute_case_sign_correct=int(np.sum(np.sign(means)==bends)),cases=len(means),prefix_score_rms=float(np.sqrt(np.mean(value[:,prefix]**2))),late_mean_scores=means.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);sources=[Path(plan['training_stimuli']),*[Path(p) for p in plan['consumed_development']]];data=[]
    for source in sources:
        r=read(source/'result.json');assert r['status']=='completed_controlled_motion_factorial' and sha256(source/'features.npz')==r['features_sha256']
        with np.load(source/'features.npz') as z:data.append({k:z[k].copy() for k in z.files})
    if 'history' in plan:
        spec=plan['history'];assert sha256(spec['normalization_checkpoint'])==spec['normalization_checkpoint_sha256'];state=torch.load(spec['normalization_checkpoint'],map_location='cpu',weights_only=True)['motion_port']
        for d in data:
            innovation=[];temporal=[]
            for k in range(len(d['motion'])):
                history=FineMotionHistory(state['reference'].numpy(),state['scale'].numpy(),fast_seconds=spec['fast_seconds'],slow_seconds=spec['slow_seconds']);episode=f'probe-case-{k}';history.begin_episode(episode);outputs=[history.step(value,episode_id=episode,decision_time=float(d['camera_receipt'][i]+.005),camera_receipt=float(d['camera_receipt'][i]),render_lowers=tuple(d['render_lowers'][i])) for i,value in enumerate(d['motion'][k])];innovation.append(np.asarray([v['innovation'] for v in outputs]));temporal.append(np.asarray([v['fast_minus_slow'] for v in outputs]))
            d['motion_innovation']=np.asarray(innovation);d['motion_fastslow']=np.asarray(temporal)
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),plan=plan,plan_sha256=sha256(a.plan),source_result_sha256=[sha256(p/'result.json') for p in sources],seed=None,actor_qualified=False,vehicle_control_sent=False,actor_optimizer_updates=0,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py');prefix=plan['prefix_frames'];late=plan['late_frames'];selected=prefix+late;d=data[0];target=np.zeros((len(d['case_bend']),len(selected)));target[:,len(prefix):]=d['case_bend'][:,None];saved={};reports=[]
    for representation in plan['representations']:
        features=d[representation].astype(float);fit=fit_probe(features[:,selected].reshape(-1,features.shape[-1]),target.reshape(-1),plan['ridge'],plan['normalization_floor'])
        for key,value in fit.items():saved[representation+'_'+key]=value
        evaluated=[]
        for role,source,x in zip(['train','consumed_development_0','consumed_development_1'],sources,data):
            value=scores(x[representation].astype(float),fit);saved[representation+'_'+role+'_scores']=value;evaluated.append(dict(role=role,source=str(source),**report(value,x['case_bend'],late,prefix)))
        folds=[]
        for lateral in sorted(set(d['case_lateral_m'])):
            test=d['case_lateral_m']==lateral;train=~test;local=fit_probe(features[train][:,selected].reshape(-1,features.shape[-1]),target[train].reshape(-1),plan['ridge'],plan['normalization_floor']);value=scores(features[test],local);folds.append(dict(held_lateral_m=float(lateral),**report(value,d['case_bend'][test],late,prefix)))
        reports.append(dict(representation=representation,features=features.shape[-1],evaluations=evaluated,train_leave_position_out=folds))
    np.savez_compressed(out/'probes.npz',**saved);write_json(out/'result.json',dict(status='completed_controlled_bend_information_probe',reports=reports,probe_arrays_sha256=sha256(out/'probes.npz'),actor_optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,new_track_generalization_verified=False,termination_reason='Comparison-only TRAIN ridge probes plus TRAIN leave-position-out and explicitly consumed-development evaluation; no model/circuit/actor changes or physical corner understanding claim'))


if __name__=='__main__':main()
