"""Independent state-index/aggregation/held-position probe formula checks."""
import argparse,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_controlled_motion_function import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--probes',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.record);r=read(root/'result.json');m=read(root/'manifest.json')['config'];assert r['status']=='completed_controlled_flight_stage_recording' and sha256(root/'states.npz')==r['states_sha256'] and sha256(root/'neurons.json')==r['neurons_sha256'];components,motion,parent=load_readonly_motion_candidate(read(m['plan']['candidate_plan']));assert parent==m['candidate'] and r['all_component_state_dict_sha256']==[digest(v.state_dict()) for v in [*components[:5],motion]];b=components[0].motor.backbone;matrix=b.circuit.matrix.coalesce();idx=matrix.indices().numpy();nz=matrix.values().numpy()!=0;columns=np.asarray(m['state_columns']);actual_groups={key:columns[value] for key,value in m['stage_columns'].items()};motor=np.unique(np.concatenate([v for v in m['motor_group_state_indices'].values()]));external=np.concatenate([b.vision_ports.numpy(),b.body_ports.numpy(),b.steering_ports.numpy(),b.goal_ports.numpy()]);expected_pre=np.setdiff1d(np.unique(idx[1,nz & np.isin(idx[0],motor)]),np.concatenate([external,motor]));np.testing.assert_array_equal(expected_pre,actual_groups['presynaptic_candidate_state']);np.testing.assert_array_equal(motor,actual_groups['individual_motor_state']);data={};max_pool=0.;max_reference=0.
    with np.load(root/'states.npz') as z:
        for name in ('visual_logits','fine_logits','visual_current','muscles'):data['six_muscles' if name=='muscles' else name]=[z[f'source_{j}_{name}'].astype(float) for j in range(4)]
        for name,cols in m['stage_columns'].items():data[name]=[z[f'source_{j}_state'][...,cols].astype(float) for j in range(4)]
        for j,spec in enumerate(m['plan']['sources']):
            state=z[f'source_{j}_state'];np.testing.assert_array_equal(state[::2,:12],state[1::2,:12]);mu=z[f'source_{j}_muscles'];pooled=np.stack([state[...,np.searchsorted(columns,group)].astype(float).mean(-1) for group in m['motor_group_state_indices'].values()],axis=-1);max_pool=max(max_pool,float(abs(mu-pooled).max()));assert max_pool<1e-8
            reference=Path(spec['response']);rr=read(reference/'result.json');assert sha256(reference/'responses.npz')==rr['responses_sha256']
            with np.load(reference/'responses.npz') as ref:expected=np.concatenate([ref[f'group_{3*k+1}_fine_prediction'] for k in range(6)])
            max_reference=max(max_reference,float(abs(z[f'source_{j}_prediction']-expected).max()));assert max_reference<=2e-6
    for key,name in [('scene','pooled_scene'),('motion','fine_motion')]:
        data[name]=[]
        for spec in m['plan']['sources']:
            with np.load(Path(spec['stimuli'])/'features.npz') as z:data[name].append(z[key].astype(float))
    data['effective_visual_logits']=[np.log(np.clip(v,1e-7,1-1e-7)/(1-np.clip(v,1e-7,1-1e-7))) for v in data['visual_current']];data['motor_group_residual']=[]
    for original in data['individual_motor_state']:
        residual=original.copy()
        for group in m['motor_group_state_indices'].values():
            cols=np.searchsorted(motor,group);residual[...,cols]-=original[...,cols].mean(-1,keepdims=True)
        data['motor_group_residual'].append(residual)
    fit_count=0;maximum_score=0.;maximum_normal_residual=0.;allcases=np.arange(12);audits=[]
    for name in a.probes:
        probe=Path(name);pr=read(probe/'result.json');pm=read(probe/'manifest.json')['config'];assert pr['status']=='completed_controlled_flight_stage_information_probe' and pm['record_result_sha256']==sha256(root/'result.json') and sha256(probe/'probes.npz')==pr['probes_sha256'];ridge=pm['ridge'];source_report=[]
        with np.load(probe/'probes.npz') as z:
            for report in pr['reports']:
                stage,form=report['stage'],report['form'];key=f'{stage}_{form}';xs=data[stage] if form=='raw' else [x-x[:,10:11] for x in data[stage]]
                for fold in (None,0,1,2):
                    cases=allcases if fold is None else np.setdiff1d(allcases,[2*fold,2*fold+1,6+2*fold,7+2*fold]);base=key if fold is None else f'{key}_fold_{fold}';reference,scale,weight,bias=[z[f'{base}_{k}'] for k in ('reference','scale','weight','bias')];train=np.concatenate([np.concatenate([x[cases,:11].reshape(-1,x.shape[-1]),x[cases,25:].reshape(-1,x.shape[-1])]) for x in xs[:3]]);target=np.tile(np.concatenate([np.zeros(len(cases)*11),np.repeat(np.array([-1.,1.]*6)[cases],16)]),3);std=train.std(0);floor=max(float(np.median(std))*1e-4,1e-8);np.testing.assert_allclose(reference,train.mean(0),atol=1e-12,rtol=1e-12);np.testing.assert_allclose(scale,np.maximum(std,floor),atol=1e-12,rtol=1e-12);f=np.clip((train-reference)/scale,-5,5);center=f-f.mean(0);residual=center.T@(center@weight-(target-target.mean()))+ridge*weight;ratio=float(np.linalg.norm(residual)/max(np.linalg.norm(center.T@(target-target.mean())),1e-12));maximum_normal_residual=max(maximum_normal_residual,ratio);assert ratio<1e-5;fit_count+=1
                    for j in range(4 if fold is None else 3):
                        score=np.clip((xs[j]-reference)/scale,-5,5)@weight+bias;stored=z[f'{base}_source_{j}_scores'];maximum_score=max(maximum_score,float(abs(score-stored).max()));assert maximum_score<1e-10;selected=allcases if fold is None else np.setdiff1d(allcases,cases);correct=int((score[selected,25:].mean(1)*np.array([-1.,1.]*6)[selected]>0).sum());record=report['evaluations'][j] if fold is None else next(v for v in report['held_lateral_cv'] if v['excluded_lateral']==fold and v['source']==j);assert record['absolute_sign_correct']==correct and record['cases']==len(selected)
                        if fold is not None:assert record['train_cases']==cases.tolist() and record['excluded_cases']==selected.tolist()
                source_report.append(dict(stage=stage,form=form,dimensions=report['dimensions'],held_lateral_absolute_correct=report['held_lateral_absolute_correct'],held_lateral_cases=report['held_lateral_cases']))
        audits.append(dict(probe=str(probe),result_sha256=sha256(probe/'result.json'),reports=source_report))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),record_result_sha256=sha256(root/'result.json'),actor_qualified=False,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_controlled_flight_stage_audit',fixed_parent_components_verified=True,presynaptic_neurons_verified=len(expected_pre),individual_motor_neurons_verified=len(motor),motor_to_six_aggregation_max_difference=max_pool,original_reference_prediction_max_difference=max_reference,common_neural_prefix_including_first_branch_output_verified=True,comparison_fits_verified=fit_count,held_lateral_normalization_and_exclusion_verified=True,maximum_normal_equation_relative_residual=maximum_normal_residual,maximum_recomputed_score_difference=maximum_score,probes=audits,full_neural_replay_by_audit=False,actor_qualified=False,vehicle_control_sent=False,physiological_function_verified=False,actual_driving_improvement_verified=False,termination_reason='Independent graph-column selection, motor aggregation, source/reference/prefix checks, TRAIN-only fit normal equations, excluded-position normalizers and stored score/count formulas. Probe dimensionality differs and no information loss rate or actor qualification follows.'))

if __name__=='__main__':main()
