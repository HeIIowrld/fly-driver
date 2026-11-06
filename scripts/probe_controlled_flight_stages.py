"""Comparison-only internal stage probes with TRAIN lateral-position exclusions."""
import argparse,shutil,time
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def training_samples(x,cases):
    v=x[cases];return np.concatenate([v[:,:11].reshape(-1,v.shape[-1]),v[:,25:].reshape(-1,v.shape[-1])]),np.concatenate([np.zeros(len(cases)*11),np.repeat(np.asarray([-1.,1.]*6)[cases],16)])


def fit(xs,cases,ridge):
    data=[training_samples(x,cases) for x in xs[:3]];x=np.concatenate([v[0] for v in data]);y=np.concatenate([v[1] for v in data]);reference=x.mean(0);std=x.std(0);floor=max(float(np.median(std))*1e-4,1e-8);scale=np.maximum(std,floor);f=np.clip((x-reference)/scale,-5,5);mean=f.mean(0);center=f-mean;target_mean=y.mean();target=y-target_mean
    if f.shape[1]<=len(f):w=np.linalg.solve(center.T@center+ridge*np.eye(f.shape[1]),center.T@target)
    else:w=center.T@np.linalg.solve(center@center.T+ridge*np.eye(len(f)),target)
    return reference,scale,w,float(target_mean-mean@w),floor


def score(x,model):
    reference,scale,w,bias,_=model;return np.clip((x-reference)/scale,-5,5)@w+bias


def stats(value,cases):
    late=value[cases,25:].mean(1);sign=np.asarray([-1,1]*6)[cases];pairs=[k for k in range(0,12,2) if k in cases and k+1 in cases];margin=[float((value[k+1,25:].mean()-value[k,25:].mean())/2) for k in pairs];return dict(absolute_sign_correct=int((late*sign>0).sum()),cases=len(cases),pair_ordering_positive=sum(v>0 for v in margin),pairs=len(pairs),pair_margins=margin,prefix_score_rms=float(np.sqrt((value[cases,:11]**2).mean())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--output',required=True);p.add_argument('--ridge',type=float,default=.01);p.add_argument('--stages',nargs='+');a=p.parse_args();root=Path(a.record);r=read(root/'result.json');m=read(root/'manifest.json')['config'];assert r['status']=='completed_controlled_flight_stage_recording' and sha256(root/'states.npz')==r['states_sha256'];data={};start=time.perf_counter();saved={};reports=[]
    with np.load(root/'states.npz') as z:
        for name in ('visual_logits','fine_logits','visual_current','muscles'):data['six_muscles' if name=='muscles' else name]=[z[f'source_{j}_{name}'].astype(float) for j in range(4)]
        for name,columns in m['stage_columns'].items():data[name]=[z[f'source_{j}_state'][...,columns].astype(float) for j in range(4)]
    for key,name in [('scene','pooled_scene'),('motion','fine_motion')]:
        data[name]=[]
        for source in m['plan']['sources']:
            with np.load(Path(source['stimuli'])/'features.npz') as z:data[name].append(z[key].astype(float))
    data['effective_visual_logits']=[np.log(np.clip(v,1e-7,1-1e-7)/(1-np.clip(v,1e-7,1-1e-7))) for v in data['visual_current']]
    motor_global=np.asarray(m['state_columns'])[m['stage_columns']['individual_motor_state']];data['motor_group_residual']=[]
    for original in data['individual_motor_state']:
        residual=original.copy()
        for members in m['motor_group_state_indices'].values():
            cols=np.searchsorted(motor_global,members);assert np.array_equal(motor_global[cols],members);residual[...,cols]-=original[...,cols].mean(-1,keepdims=True)
        data['motor_group_residual'].append(residual)
    if a.stages:
        assert set(a.stages)<=set(data);data={key:value for key,value in data.items() if key in a.stages}
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),record_result_sha256=sha256(root/'result.json'),source_roles=[s['current_role'] for s in m['plan']['sources']],ridge=a.ridge,seed=None,actor_optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__,prefix_reference='Frame10 of declared common straight prefix; diagnostic reference timing known from generator, not a deployed corner onset detector'));shutil.copyfile(__file__,out/'executed-entrypoint.py');all_cases=np.arange(12)
    for stage,xs_raw in data.items():
        for form in ('raw','prefix_relative'):
            xs=xs_raw if form=='raw' else [x-x[:,10:11] for x in xs_raw];model=fit(xs,all_cases,a.ridge);key=f'{stage}_{form}';scores=[score(x,model) for x in xs];evaluations=[dict(source=j,role='consumed_development' if j==3 else 'declared_train',**stats(v,all_cases)) for j,v in enumerate(scores)];cv=[]
            for lateral in range(3):
                excluded=np.array([2*lateral,2*lateral+1,6+2*lateral,7+2*lateral]);included=np.setdiff1d(all_cases,excluded);fold=fit(xs,included,a.ridge)
                for j in range(3):
                    values=score(xs[j],fold);saved[f'{key}_fold_{lateral}_source_{j}_scores']=values;cv.append(dict(excluded_lateral=lateral,source=j,train_cases=included.tolist(),excluded_cases=excluded.tolist(),**stats(values,excluded)))
                for name,v in zip(('reference','scale','weight','bias'),fold[:4]):saved[f'{key}_fold_{lateral}_{name}']=np.asarray(v)
            for j,value in enumerate(scores):saved[f'{key}_source_{j}_scores']=value
            for name,v in zip(('reference','scale','weight','bias'),model[:4]):saved[f'{key}_{name}']=np.asarray(v)
            reports.append(dict(stage=stage,form=form,dimensions=xs[0].shape[-1],scale_floor=model[4],evaluations=evaluations,held_lateral_cv=cv,held_lateral_absolute_correct=sum(v['absolute_sign_correct'] for v in cv),held_lateral_cases=sum(v['cases'] for v in cv)));write_json(out/'progress.json',dict(stage_forms=len(reports),total=2*len(data),elapsed_seconds=time.perf_counter()-start))
    np.savez_compressed(out/'probes.npz',**saved);write_json(out/'result.json',dict(status='completed_controlled_flight_stage_information_probe',reports=reports,probes_sha256=sha256(out/'probes.npz'),comparison_ridge_fits=4*len(reports),actor_optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,actual_driving_improvement_verified=False,physiological_function_verified=False,full_stage_information_loss_rate_claimed=False,termination_reason='Same examples and normalization/ridge family across stages; three TRAIN lateral exclusions and already consumed fourth-source development. Stage dimensionality/capacity differ. Prefix reference is declared diagnostic timing. No new actor or physical driving qualification.'))

if __name__=='__main__':main()
