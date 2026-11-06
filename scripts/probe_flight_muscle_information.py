"""Comparison-only bend information after the frozen measured flight graph.

Existing consumed synthetic sources; no actor training or deployment.
"""
import argparse,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def mode(x,representation):
    if representation=='six_muscles':return x
    if representation=='bilateral_opponents':return np.stack([x[...,1]-x[...,0],x[...,3]-x[...,2]],axis=-1)
    return np.stack([(x[...,1]+x[...,0])/2,(x[...,3]+x[...,2])/2,(x[...,5]+x[...,4])/2],axis=-1)


def source(root):
    root=Path(root);r=read(root/'result.json');m=read(root/'manifest.json')['config'];assert r['status']=='completed_controlled_motion_factorial_response' and sha256(root/'responses.npz')==r['responses_sha256'];sm=read(Path(m['arguments']['stimuli'])/'manifest.json')['config'];arrays=[]
    with np.load(root/'responses.npz') as z:
        for pair in range(6):arrays.extend(z[f'group_{3*pair+1}_fine_muscles'])
    x=np.asarray(arrays,dtype=float);assert x.shape==(12,41,6);return x,m['candidate'],sm['plan']['texture_seed']


def samples(x):
    prefix=x[:,:11].reshape(-1,x.shape[-1]);late=x[:,25:].reshape(-1,x.shape[-1]);y=np.concatenate([np.zeros(len(prefix)),np.repeat(np.tile([-1.,1.],6),16)]);return np.concatenate([prefix,late]),y


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--ridge',type=float,default=.01);a=p.parse_args();assert a.ridge>0;roots=['runs/controlled-visual-factorial-20261104/responses-01','runs/controlled-visual-factorial-20261104/test-parent-01','runs/visual-route-protection-20261104/reserved-parent-01','runs/motion-descending-transfer-20261104/reserved-parent-01'];data=[source(root) for root in roots];assert all(d[1]==data[0][1] for d in data);reports=[];saved={};out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),sources=[dict(response=root,result_sha256=sha256(Path(root)/'result.json'),texture_seed=d[2],role='promoted_train_this_comparison' if i<3 else 'already_consumed_development_no_fresh_test') for i,(root,d) in enumerate(zip(roots,data))],candidate=data[0][1],seed=None,actor_optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py')
    for arm in ('repeated','diverse'):
        for representation in ('six_muscles','bilateral_opponents','bilateral_common'):
            transformed=[mode(d[0],representation) for d in data];training=[samples(transformed[j if arm=='diverse' else 0]) for j in range(3)];x=np.concatenate([z[0] for z in training]);y=np.concatenate([z[1] for z in training]);reference=x.mean(0);scale=np.maximum(x.std(0),1e-8);features=np.clip((x-reference)/scale,-5,5);mean=features.mean(0);target_mean=y.mean();center=features-mean;w=np.linalg.solve(center.T@center+a.ridge*np.eye(x.shape[-1]),center.T@(y-target_mean));bias=target_mean-mean@w;saved[f'{arm}_{representation}_weight']=w;saved[f'{arm}_{representation}_reference']=reference;saved[f'{arm}_{representation}_scale']=scale;saved[f'{arm}_{representation}_bias']=np.asarray(bias);evaluations=[]
            for j,feature in enumerate(transformed):
                score=np.clip((feature-reference)/scale,-5,5)@w+bias;late=score[:,25:].mean(1);margins=(late[1::2]-late[::2])/2;correct=int((late*np.tile([-1,1],6)>0).sum());saved[f'{arm}_{representation}_source_{j}_score']=score;evaluations.append(dict(source=j,texture_seed=data[j][2],role='consumed_development' if j==3 else 'declared_train_cohort',positive_pair_count=int((margins>0).sum()),pair_count=6,absolute_case_sign_correct=correct,cases=12,late_scores=late.tolist(),prefix_score_rms=float(np.sqrt((score[:,:11]**2).mean())),signed_pair_margins=margins.tolist()))
            reports.append(dict(arm=arm,representation=representation,features=x.shape[-1],evaluations=evaluations))
    np.savez_compressed(out/'probes.npz',**saved);write_json(out/'result.json',dict(status='completed_comparison_flight_muscle_information_probe',reports=reports,probes_sha256=sha256(out/'probes.npz'),actor_optimizer_updates=0,comparison_ridge_fits=6,vehicle_control_sent=False,actor_qualified=False,physiological_function_verified=False,actual_driving_improvement_verified=False,termination_reason='Matched three-source versus repeated-source comparison ridge, prefix target0 and late bend signs, evaluated only already consumed synthetic conditions at body80. Learned readouts are comparison-only, not actors or new flight circuits.'))

if __name__=='__main__':main()
