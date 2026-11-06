"""Frozen flight circuit: temporally matched rotation context and named interventions.

Known analytic body channels only; missing forces/slips/pressure/wheel rates stay
masked. No measured physiology/native sensing/driving qualification or controls.
"""
import argparse,gc,hashlib,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.synthetic_yaw_body import synthetic_turn_body,yaw_profile
from flybrain_drive.temporal_port_gain import load_temporal_gain
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.audit_controlled_motion_function import digest
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.sac_steering_experiment import snapshot,restore

ARMS=('normal','context_off','context_reverse','context_delayed120')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stimuli',required=True);p.add_argument('--checkpoint');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);revision=subprocess.check_output(['git','-c','safe.directory='+str(Path.cwd()).replace('\\','/'),'rev-parse','HEAD'],text=True).strip();source=Path(a.stimuli);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];plan=sm['plan'];assert all(sha256(path)==value for path,value in plan['source_sha256'].items());assert sr['status']=='completed_visual_body_rotation_stimuli' and sha256(source/'features.npz')==sr['features_sha256'] and sha256(plan['candidate_plan'])==plan['candidate_plan_sha256'];components,motion,receipt=load_readonly_motion_candidate(read(plan['candidate_plan']));model,head,dn,optical,body_port,_=components;gain_handle=None
    if a.checkpoint:receipt,gain_handle=load_temporal_gain(a.checkpoint,components,motion,receipt)
    for module in [*components[:5],motion]:module.eval().requires_grad_(False)
    protected=[digest(v.state_dict()) for v in [*components[:5],motion]]
    with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),source_revision_observed_before_execution=revision,stimuli_result_sha256=sha256(source/'result.json'),features_sha256=sr['features_sha256'],candidate=receipt,arms=ARMS,scope=__doc__,seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');holder={};stages={};hooks=[];saved={};reports=[];goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);contract=sha256('configs/ac-virtual-navigation-body-v2.json');b=model.motor.backbone;motor=torch.unique(torch.cat([b.circuit.output_ports[getattr(b.readout,k)] for k in ('b1_L','b1_R','b2_L','b2_R','power_L','power_R')]));start=time.perf_counter();before=gc.isenabled();gc.disable();result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False)
    def body_at(t):
        case=holder['case_metadata'];return synthetic_turn_body(t-100.,case['speed_mps'],case['yaw_rate_target_rad_s'],holder['temperature'],onset=plan['rotation_onset_s'],duration=plan['rotation_duration_s'])
    def descending(module,args,value):
        k,i=holder['case'],holder['frame'];steer,power=value;return steer+dn(torch.tensor(x['static'][k,i:i+1])),power
    def visual(module,args,value):
        k,i=holder['case'],holder['frame'];return value+optical(torch.tensor(x['static'][k,i:i+1,64:]))+motion(torch.tensor(x['motion'][k,i:i+1]))
    def body(module,args,value):return value+body_port(holder['extension'],holder['extension_valid'])
    def context(module,args,value):
        traces,history,valid=args;stages['raw_context']=value[0].numpy().copy();effective=value;effective_yaw=history[0,:,4].clone();arm=holder['arm'] if holder['frame']>=plan['branch_frame'] else 'normal'
        if arm=='context_off':effective=torch.zeros_like(value)
        elif arm in ('context_reverse','context_delayed120'):
            changed=history.clone()
            if arm=='context_reverse':changed[:,:,4]*=-1
            else:
                case=holder['case_metadata'];rates=yaw_profile(x['view_body_sample_time'][holder['frame']]-100.-.12,case['yaw_rate_target_rad_s'],onset=plan['rotation_onset_s'],duration=plan['rotation_duration_s'])[0];changed[0,:,4]=torch.tensor(rates,dtype=history.dtype)
            effective_yaw=changed[0,:,4].clone();effective=module.forward(traces,changed,valid)
        stages['effective_context']=effective[0].numpy().copy();stages['effective_context_yaw']=effective_yaw.numpy().copy();return effective
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body),model.view_context.register_forward_hook(context)]
    def step(k,i):
        holder.update(case=k,frame=i,case_metadata=sm['cases'][k]);clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;sample=float(x['decision_body_sample_time'][i]);values,valid,extra,extra_valid=body_at(sample);holder.update(extension=torch.tensor(extra[None],dtype=torch.float32),extension_valid=torch.tensor(extra_valid[None]));current=NavigationBodyObservation(holder['episode'],5*i+4,sample,now,values.copy(),valid.copy(),contract);history=[]
        for j,t in enumerate(x['view_body_sample_time'][i]):
            value,mask,_,_=body_at(float(t));history.append(NavigationBodyObservation(holder['episode'],5*i+j,float(t),camera,value.copy(),mask.copy(),contract))
        ctx=per_view_temporal_body_context(tuple(history),current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now)
        with torch.no_grad():
            _,muscles=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=torch.tensor(x['scene'][k,i:i+1]),scene_valid=views,body_context=ctx);prediction=head(muscles)[0].numpy().copy()
        return dict(prediction=prediction,muscles=muscles[0].numpy().copy(),raw_context=stages['raw_context'].copy(),effective_context=stages['effective_context'].copy(),effective_context_yaw=stages['effective_context_yaw'].copy(),body_ports=model.previous_drive[0,b.body_ports].numpy().copy(),descending_ports=model.previous_drive[0,b.steering_ports].numpy().copy(),visual_current=model.previous_drive[0,b.vision_ports].numpy().copy(),visual_neurons=model.state[0,b.vision_ports].numpy().copy(),descending_neurons=model.state[0,b.steering_ports].numpy().copy(),motor_neurons=model.state[0,motor].numpy().copy())
    try:
        torch.set_flush_denormal(True);write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(model))
        for pair in range(0,len(sm['cases']),2):
            right,left=sm['cases'][pair:pair+2];assert right['bend']==-1 and left['bend']==1 and all(right[key]==left[key] for key in ('speed_mps','lateral_m','yaw_rate_target_rad_s'))
            for temperature in plan['temperature_c']:
                group=len(reports);holder.update(arm='normal',temperature=temperature,episode=f'synthetic-rotation-group-{group}');model.reset_episode();prefix=[step(pair,i) for i in range(plan['branch_frame'])];state=snapshot(model);state_sha=state_digest(state);branches={}
                for arm in ARMS:
                    holder['arm']=arm;output=[]
                    for k in (pair,pair+1):
                        restore(model,state);assert state_digest(snapshot(model))==state_sha;sequence=prefix+[step(k,i) for i in range(plan['branch_frame'],plan['frame_count'])];output.append(sequence)
                    for key in output[0][0]:saved[f'group_{group}_{arm}_{key}']=np.asarray([[v[key] for v in branch] for branch in output])
                    prediction=saved[f'group_{group}_{arm}_prediction'];reference=saved[f'group_{group}_normal_prediction'];late=prediction[:,plan['evaluation_first_frame']:];margin=float(plan['expected_canonical_steering_sign_per_left_bend']*np.mean(late[1,:,0]-late[0,:,0])/2);branches[arm]=dict(hypothesized_bend_margin=margin,direction_positive=margin>plan['bend_ordering_minimum_margin'],post_prefix_max_abs_action_change_vs_normal=np.max(abs(prediction[:,plan['branch_frame']:]-reference[:,plan['branch_frame']:]),axis=(0,1)).tolist(),context_post_prefix_norm_max=float(np.linalg.norm(saved[f'group_{group}_{arm}_effective_context'][:,plan['branch_frame']:],axis=-1).max()))
                for label,times in [('decision',x['decision_body_sample_time']),('views',x['view_body_sample_time'])]:
                    flat=[body_at(float(t)) for t in times.reshape(-1)];shape=times.shape
                    for j,name in enumerate(('body21','body21_valid','extension19','extension19_valid')):saved[f'group_{group}_{label}_{name}']=np.asarray([v[j] for v in flat]).reshape(*shape,-1)
                reports.append(dict(group=group,right_case=pair,left_case=pair+1,temperature_c=temperature,speed_mps=right['speed_mps'],yaw_rate_target_rad_s=right['yaw_rate_target_rad_s'],initial_neural_state_sha256=state_sha,branches=branches));write_json(out/'progress.json',dict(completed_groups=len(reports),total=len(sm['cases'])//2*len(plan['temperature_c']),elapsed_seconds=time.perf_counter()-start))
        assert protected==[digest(v.state_dict()) for v in [*components[:5],motion]];np.savez_compressed(out/'responses.npz',**saved);write_json(out/'groups.json',reports);zero=max(float(abs(saved[f"group_{g['group']}_{arm}_prediction"]-saved[f"group_{g['group']}_normal_prediction"]).max()) for g in reports if g['yaw_rate_target_rad_s']==0 for arm in ARMS);assert zero<=2e-6;result.update(status='completed_visual_body_rotation_context_response',groups=len(reports),zero_rate_intervention_prediction_max_difference=zero,protected_component_sha256=protected,responses_sha256=sha256(out/'responses.npz'),groups_sha256=sha256(out/'groups.json'),synthetic_body_only=True,actual_native_samples=False,physiological_correctness_verified=False,actual_driving_improvement_verified=False,termination_reason='Same body/time/full neural prefix within road bend pairs and context interventions; correct history, context0, reversed/delayed historical yaw values with raw body and other ports retained. Known channels only and unknowns masked. No teacher geometry actor input, fitting or vehicle control.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_groups=len(reports))
    finally:
        for hook in hooks:hook.remove()
        if gain_handle is not None:gain_handle.remove()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
