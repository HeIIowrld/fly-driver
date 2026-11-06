"""Fixed candidate response to synthetic factorial vision with same body/prefix.

Synthetic body features and temperatures are hypotheses, not measured native
sensors or evidence of temperature adaptation. No controls or training.
"""
import argparse,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.functional_motion_overlay import load_functional_motion_overlay
from flybrain_drive.temporal_port_gain import load_temporal_gain
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.sac_steering_experiment import snapshot,restore


def synthetic_body(speed,temperature,native_reference):
    values=np.zeros(21);values[8:12]=.25;values[12:16]=temperature;values[16]=speed;values[17:21]=speed/.29
    mask=np.array([True,True,False,False,True,False,False,False]+[True]*13)
    native=native_reference.copy();native[:2]=0;native[3:7]=0;native[11:15]=temperature
    return values,mask,native,np.ones(19,dtype=bool)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stimuli',required=True);p.add_argument('--experimental-checkpoint');p.add_argument('--route-checkpoint');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);source=Path(a.stimuli);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];plan=sm['plan']
    if sr['status']!='completed_controlled_motion_factorial' or sha256(source/'features.npz')!=sr['features_sha256'] or sha256(plan['candidate_plan'])!=plan['candidate_plan_sha256']:raise ValueError('Completed frozen synthetic stimuli/candidate required')
    if sha256(plan['steering_sign_evidence'])!=plan['steering_sign_evidence_sha256']:raise ValueError('Historical sign basis changed')
    components,motion,receipt=load_readonly_motion_candidate(read(plan['candidate_plan']));model,head,dn,optical,body_port,_=components
    if motion is None:raise ValueError('Audited motion candidate required')
    if a.experimental_checkpoint and a.route_checkpoint:raise ValueError('Exactly one research overlay allowed')
    if a.experimental_checkpoint:receipt=load_functional_motion_overlay(a.experimental_checkpoint,components,motion,receipt)
    gain_handle=None
    if a.route_checkpoint:receipt,gain_handle=load_temporal_gain(a.route_checkpoint,components,motion,receipt)
    for component in components[:5]:component.eval().requires_grad_(False)
    with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),stimuli_result_sha256=sha256(source/'result.json'),features_sha256=sr['features_sha256'],plan=plan,candidate=receipt,seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py');goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);holder=dict(case=0,frame=0,arm='fine');stages={};reports=[];saved={};hooks=[];before=gc.isenabled();gc.disable();start=time.perf_counter();result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);contract=sha256('configs/ac-virtual-navigation-body-v2.json')
    def descending(module,args,value):
        k,i=holder['case'],holder['frame'];steer,power=value;return steer+dn(torch.tensor(x['static'][k,i:i+1])),power
    def visual(module,args,value):
        k,i=holder['case'],holder['frame'];fine=motion(torch.tensor(x['motion'][k,i:i+1]));total=value+optical(torch.tensor(x['static'][k,i:i+1,64:]))
        if holder['arm']=='fine':total=total+fine
        stages['motion_delta']=fine[0].numpy().copy();stages['visual_ports']=total[0].numpy().copy();return total
    def body(module,args,value):return value+body_port(holder['native'],holder['native_valid'])
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)]
    def step(case,i):
        holder.update(case=case,frame=i);clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005
        current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,holder['body'].copy(),holder['mask'].copy(),contract)
        history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,holder['body'].copy(),holder['mask'].copy(),contract) for j,t in enumerate(clocks))
        ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now)
        with torch.no_grad():_,muscles=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=torch.tensor(x['scene'][case,i:i+1]),scene_valid=views,body_context=ctx);prediction=head(muscles)[0].numpy().copy()
        return prediction,muscles[0].numpy().copy(),stages['motion_delta'].copy(),stages['visual_ports'].copy()
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Fixed CPU numerical mode unavailable')
        write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(model))
        for pair in range(0,len(sm['cases']),2):
            right,left=pair,pair+1;rc,lc=sm['cases'][right],sm['cases'][left]
            if rc['bend']!=-1 or lc['bend']!=1 or rc['lateral_m']!=lc['lateral_m'] or rc['speed_mps']!=lc['speed_mps']:raise ValueError('Sealed opposite pair ordering differs')
            for temperature in plan['temperature_c']:
                body_values,mask,native,valid=synthetic_body(rc['speed_mps'],temperature,body_port.reference.numpy());holder.update(body=body_values,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),arm='fine');body_digest=hashlib.sha256(body_values.tobytes()+mask.tobytes()+native.tobytes()+valid.tobytes()).hexdigest();model.reset_episode();prefix=[step(right,i) for i in range(plan['branch_frame'])];state=snapshot(model);initial=state_digest(state);group=len(reports);branches={}
                for arm in ('fine','motion_off_after_prefix'):
                    holder['arm']='fine' if arm=='fine' else 'off';outputs=[]
                    for case in (right,left):
                        restore(model,state)
                        if state_digest(snapshot(model))!=initial:raise ValueError('Paired branch starting neural state differs')
                        sequence=prefix+[step(case,i) for i in range(plan['branch_frame'],plan['frame_count'])];outputs.append(sequence)
                    for j,key in enumerate(('prediction','muscles','motion_delta','visual_ports')):saved[f'group_{group}_{arm}_{key}']=np.asarray([[value[j] for value in sequence] for sequence in outputs])
                    prediction=saved[f'group_{group}_{arm}_prediction'];late=prediction[:,plan['evaluation_first_frame']:];margin=float(plan['expected_canonical_steering_sign_per_left_bend']*np.mean(late[1,:,0]-late[0,:,0])/2);branches[arm]=dict(hypothesized_bend_margin=margin,hypothesized_bend_direction_positive=margin>plan['bend_ordering_minimum_margin'],late_mean_prediction=late.mean(1).tolist(),maximum_absolute_bend_steering_difference=float(abs(late[1,:,0]-late[0,:,0]).max()))
                if body_digest!=hashlib.sha256(body_values.tobytes()+mask.tobytes()+native.tobytes()+valid.tobytes()).hexdigest():raise ValueError('Fixed synthetic body changed inside pair')
                saved[f'group_{group}_body21']=body_values;saved[f'group_{group}_body21_valid']=mask;saved[f'group_{group}_native19']=native;saved[f'group_{group}_native19_valid']=valid;reports.append(dict(group=group,right_case=right,left_case=left,lateral_m=rc['lateral_m'],speed_mps=rc['speed_mps'],temperature_c=temperature,initial_neural_state_sha256=initial,fixed_body_sha256=body_digest,branches=branches));write_json(out/'progress.json',dict(completed_groups=len(reports),total=len(sm['cases'])//2*len(plan['temperature_c']),elapsed_seconds=time.perf_counter()-start))
        np.savez_compressed(out/'responses.npz',**saved);write_json(out/'groups.json',reports);counts={arm:sum(g['branches'][arm]['hypothesized_bend_direction_positive'] for g in reports) for arm in ('fine','motion_off_after_prefix')};result.update(status='completed_controlled_motion_factorial_response',groups=len(reports),opposite_pairs_per_arm=len(reports),hypothesized_positive_bend_pairs=counts,groups_sha256=sha256(out/'groups.json'),responses_sha256=sha256(out/'responses.npz'),synthetic_body_only=True,temperature_response_correctness_verified=False,physical_counterfactual_action_correctness_verified=False,new_track_generalization_verified=False,termination_reason='Frozen circuit/readout on synthetic factorial; same complete neural prefix and body within bend/ablation branches, hypothetical historical-sign ordering measured after onset transient; no actual tyres, driving or physiological qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_groups=len(reports))
    finally:
        for hook in hooks:hook.remove()
        if gain_handle is not None:gain_handle.remove()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
