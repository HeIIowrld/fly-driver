"""TRAIN-only exact flight-port adjoints, finite differences and muscle cancellation.

Port/logit perturbations are engineering interventions, not measured physiology.
No optimizer, new actor adapter, simulator requests or vehicle control.
"""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.sac_steering_experiment import snapshot,restore


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stimuli',required=True);p.add_argument('--reference-response',required=True);p.add_argument('--output',required=True);p.add_argument('--pair-count',type=int,choices=range(1,7),default=6);p.add_argument('--descending-fd-epsilon',type=float,default=.02);a=p.parse_args();assert 0<a.descending_fd_epsilon<=.02;torch.set_num_threads(1);torch.manual_seed(781114);rng=np.random.default_rng(781114);source=Path(a.stimuli);sm=read(source/'manifest.json')['config'];sr=read(source/'result.json');plan=sm['plan'];ref=Path(a.reference_response);rr=read(ref/'result.json');rm=read(ref/'manifest.json')['config']
    assert sr['status']=='completed_controlled_motion_factorial' and sha256(source/'features.npz')==sr['features_sha256'] and sha256(ref/'responses.npz')==rr['responses_sha256'] and rm['features_sha256']==sr['features_sha256']
    components,motion,receipt=load_readonly_motion_candidate(read(plan['candidate_plan']));model,head,dn,optical,body_port,_=components;assert rm['candidate']==receipt
    for module in [*components[:5],motion]:module.eval().requires_grad_(False)
    with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
    with np.load(ref/'responses.npz') as z:reference={k:z[k].copy() for k in z.files if k.endswith('_fine_prediction')}
    sizes=[len(model.motor.backbone.vision_ports),len(model.motor.backbone.steering_ports),len(model.motor.backbone.body_ports)];cuts=np.cumsum([0,*sizes]);names=['vision_logits','descending_current','body_logits'];ports=torch.cat([model.motor.backbone.vision_ports,model.motor.backbone.steering_ports,model.motor.backbone.body_ports]);n=int(cuts[-1]);holder={};hooks=[];goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);contract=sha256('configs/ac-virtual-navigation-body-v2.json');saved={};reports=[];result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);start=time.perf_counter();before=gc.isenabled();gc.disable()
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),features_sha256=sr['features_sha256'],candidate=receipt,seed=781114,port_sections={name:[int(cuts[j]),int(cuts[j+1])] for j,name in enumerate(names)},port_indices=ports.tolist(),finite_difference_epsilon=.02,descending_finite_difference_epsilon=a.descending_fd_epsilon,temperature_c=80.,train_only=True,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py')
    def perturb(j):return holder['epsilon'][int(cuts[j]):int(cuts[j+1])].unsqueeze(0)
    def descending(module,args,value):
        steer,power=value;k,i=holder['case'],holder['frame'];return steer+dn(torch.tensor(x['static'][k,i:i+1]))+perturb(1),power
    def visual(module,args,value):
        k,i=holder['case'],holder['frame'];return value+optical(torch.tensor(x['static'][k,i:i+1,64:]))+motion(torch.tensor(x['motion'][k,i:i+1]))+perturb(0)
    def body(module,args,value):return value+body_port(holder['native'],holder['native_valid'])+perturb(2)
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)]
    def step(case,i,learn):
        holder.update(case=case,frame=i);clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=holder['body'],holder['mask'];current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,values.copy(),mask.copy(),contract);history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,values.copy(),mask.copy(),contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now)
        with torch.set_grad_enabled(learn):
            fn=model.sensory_training_decision if learn else model.sensory_decision;_,mu=fn(decision_time=now,racing_goals=goals,scene_features=torch.tensor(x['scene'][case,i:i+1]),scene_valid=views,body_context=ctx);pred=head(mu)[0];return torch.cat([mu[0],pred]),model.previous_drive[0,ports].detach().numpy().copy()
    def branch(state,case,epsilon,learn):
        restore(model,state);holder['epsilon']=epsilon;seq=[];drives=[]
        for i in range(plan['branch_frame'],plan['frame_count']):
            value,drive=step(case,i,learn);seq.append(value);drives.append(drive)
        sequence=torch.stack(seq);return sequence,sequence[plan['evaluation_first_frame']-plan['branch_frame']:].mean(0),np.asarray(drives)
    try:
        torch.set_flush_denormal(True);write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(model))
        for pair in range(0,2*a.pair_count,2):
            case=sm['cases'][pair];values,mask,native,valid=synthetic_body(case['speed_mps'],80.,body_port.reference.numpy());holder.update(body=values,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),epsilon=torch.zeros(n));model.reset_episode();prefix=[step(pair,i,False)[0].detach().numpy() for i in range(plan['branch_frame'])];state=snapshot(model);branches=[]
            for case_id in (pair,pair+1):
                epsilon=torch.zeros(n,requires_grad=True);sequence,mean,drives=branch(state,case_id,epsilon,True);jac=torch.stack([torch.autograd.grad(mean[j],epsilon,retain_graph=True)[0] for j in range(9)]).detach().numpy();baseline=sequence.detach().numpy();del mean,sequence;checks=[]
                for section,name in enumerate(names):
                    direction=np.zeros(n,dtype=np.float32);values_d=rng.normal(size=sizes[section]);values_d/=np.linalg.norm(values_d);direction[cuts[section]:cuts[section+1]]=values_d;fd_epsilon=a.descending_fd_epsilon if section==1 else .02;delta=torch.tensor(direction*fd_epsilon);plus,pmean,_=branch(state,case_id,delta,False);minus,mmean,_=branch(state,case_id,-delta,False);numeric=(pmean-mmean).numpy()/(2*fd_epsilon);analytic=jac@direction;saved[f'case_{case_id}_{name}_direction']=direction;saved[f'case_{case_id}_{name}_finite_difference']=numeric;saved[f'case_{case_id}_{name}_adjoint_direction']=analytic;checks.append(dict(section=name,epsilon=fd_epsilon,maximum_absolute_difference=float(abs(numeric-analytic).max()),relative_l2_error=float(np.linalg.norm(numeric-analytic)/max(np.linalg.norm(numeric),np.linalg.norm(analytic),1e-12))))
                expected=reference[f'group_{3*(pair//2)+1}_fine_prediction'][case_id-pair];prediction=np.concatenate([np.asarray(prefix)[:,6:],baseline[:,6:]]);difference=float(abs(prediction-expected).max());assert difference<=2e-6;saved[f'case_{case_id}_baseline']=baseline;saved[f'case_{case_id}_jacobian']=jac;saved[f'case_{case_id}_port_drive']=drives;branches.append(baseline);reports.append(dict(case=case_id,source_geometry=sm['cases'][case_id],reference_max_difference=difference,sections={name:dict(muscle_jacobian_frobenius=float(np.linalg.norm(jac[:6,cuts[j]:cuts[j+1]])),steering_jacobian_norm=float(np.linalg.norm(jac[6,cuts[j]:cuts[j+1]])),pedal_jacobian_frobenius=float(np.linalg.norm(jac[7:,cuts[j]:cuts[j+1]]))) for j,name in enumerate(names)},finite_difference_checks=checks));write_json(out/'progress.json',dict(cases=len(reports),total=2*a.pair_count,elapsed_seconds=time.perf_counter()-start))
            late=plan['evaluation_first_frame']-plan['branch_frame'];delta_mu=(branches[1][late:,:6]-branches[0][late:,:6]).mean(0);contribution=delta_mu*(head.projection.weight[0]/head.training_scale).detach().numpy();saved[f'pair_{pair//2}_steering_logit_muscle_contributions']=contribution
        np.savez_compressed(out/'transfer.npz',**saved);write_json(out/'cases.json',reports);result.update(status='completed_train_flight_port_transfer_probe',cases=len(reports),port_sections={name:sizes[j] for j,name in enumerate(names)},baseline_reference_max_difference=max(z['reference_max_difference'] for z in reports),maximum_finite_difference_absolute_error=max(c['maximum_absolute_difference'] for z in reports for c in z['finite_difference_checks']),transfer_sha256=sha256(out/'transfer.npz'),cases_sha256=sha256(out/'cases.json'),termination_reason='Exact TRAIN synthetic body80 common prefixes; persistent post-branch small port perturbations, exact adjoints and centered finite differences through original recurrent flight circuit. Engineering unit-specific sensitivities; no optimizer/actor changes/physiology or actual vehicle qualification.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_cases=len(reports))
    finally:
        for hook in hooks:hook.remove()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
