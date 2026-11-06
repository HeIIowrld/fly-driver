"""Independent trajectory, sensor-clock, context-filter, muscle and intervention audit."""
import argparse,math,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_controlled_motion_function import digest


def profile(t,a,onset,duration):
    u=max(0.,min(t-onset,duration));angle=a*duration/math.pi*(1-math.cos(math.pi*u/duration));rate=a*math.sin(math.pi*u/duration) if 0<t-onset<duration else 0.;return rate,angle

def body_expected(times,speed,a,temp,onset,duration):
    omega=np.asarray([profile(float(t)-100.,a,onset,duration)[0] for t in times.reshape(-1)]).reshape(times.shape);body=np.zeros((*times.shape,21));known=np.zeros_like(body,dtype=bool);body[...,4]=omega;body[...,12:16]=temp;body[...,16]=speed;known[...,[0,1,4,12,13,14,15,16]]=True;extra=np.zeros((*times.shape,19));valid=np.zeros_like(extra,dtype=bool);extra[...,1]=speed*omega;extra[...,11:15]=temp;valid[...,[0,1,11,12,13,14]]=True;return body,known,extra,valid

def trace_opponents(scene,times,mean,scale,fast_tau,slow_tau):
    normalized=np.clip((scene.astype(np.float32)-mean.astype(np.float32))/scale.astype(np.float32),-5,5).reshape(-1,4,9,16)[:,:,1:].reshape(-1,512);fast=normalized[0].copy();slow=fast.copy();held=fast.copy();values=[]
    for i in range(len(scene)):
        if i:
            elapsed=float(times[i]-times[i-1]);full=int(np.floor(elapsed/.005));intervals=[.005]*full;tail=elapsed-full*.005
            if tail>1e-12:intervals.append(tail)
            if not intervals:intervals=[elapsed]
            for j,dt in enumerate(intervals):
                af=math.exp(-dt/fast_tau);aslow=math.exp(-dt/slow_tau);fast=np.float32(af)*fast+np.float32(1-af)*held;slow=np.float32(aslow)*slow+np.float32(1-aslow)*held;held=normalized[i if j==len(intervals)-1 else i-1].copy()
        motion=(fast-slow).reshape(4,2,4,16);opponent=np.stack([motion[:,:,1]-motion[:,:,0],motion[:,:,2]-motion[:,:,3]],axis=2);values.append(opponent.reshape(4,64))
    return np.asarray(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stimuli',required=True);p.add_argument('--responses',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);source=Path(a.stimuli);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];plan=sm['plan'];assert sr['status']=='completed_visual_body_rotation_stimuli' and sha256(source/'features.npz')==sr['features_sha256'];camera=read('configs/ac-csp-exterior-front-v3-retina.json');assert sha256('configs/ac-csp-exterior-front-v3-retina.json')==plan['camera_sha256'];components,motion,parent=load_readonly_motion_candidate(read(plan['candidate_plan']));model,head=components[:2];assert model.motion_forecast is None;protected=[digest(v.state_dict()) for v in [*components[:5],motion]];weight=model.view_context.projection.weight.detach().numpy().astype(float);mean=model.temporal.mean.numpy();scale=model.temporal.scale.numpy();onset,duration=plan['rotation_onset_s'],plan['rotation_duration_s'];maximum_pose=0.;maximum_body=0.;maximum_context=0.;maximum_readout=0.;minimum_entry=float('inf');reports=[]
    with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
    np.testing.assert_allclose(x['render_lowers']-x['view_body_sample_time'],.001,atol=1e-12,rtol=0);np.testing.assert_allclose(x['camera_receipt']+.005-x['decision_body_sample_time'],.001,atol=1e-12,rtol=0)
    for k,case in enumerate(sm['cases']):
        amplitude=case['yaw_rate_target_rad_s'];speed=case['speed_mps']
        for i,row in enumerate(x['render_lowers']):
            for j,clock in enumerate(row):
                t=float(clock-100.);rate,angle=profile(t,amplitude,onset,duration);points=[v for v in (onset,onset+duration) if 0<v<t];px=quad(lambda u:speed*math.cos(profile(u,amplitude,onset,duration)[1]),0,t,points=points,epsabs=1e-11,epsrel=1e-11)[0];py=quad(lambda u:speed*math.sin(profile(u,amplitude,onset,duration)[1]),0,t,points=points,epsabs=1e-11,epsrel=1e-11)[0];maximum_pose=max(maximum_pose,float(abs(x['render_pose'][k,i,j]-[px,py,angle,rate]).max()));origin=camera['views'][camera['view_order'][j]]['origin_body_m'];minimum_entry=min(minimum_entry,plan['bend_start_m']-(px+math.cos(angle)*origin[0]-math.sin(angle)*origin[1]))
        assert maximum_pose<1e-10
    assert minimum_entry>0 and abs(minimum_entry-sr['minimum_camera_bend_entry_margin_m'])<1e-10
    for name in a.responses:
        root=Path(name);r=read(root/'result.json');m=read(root/'manifest.json')['config'];assert r['status']=='completed_visual_body_rotation_context_response' and m['features_sha256']==sr['features_sha256'] and sha256(root/'responses.npz')==r['responses_sha256'] and sha256(root/'groups.json')==r['groups_sha256'] and r['protected_component_sha256']==protected;groups=read(root/'groups.json');zero_max=0.;effects=[]
        with np.load(root/'responses.npz') as z:
            for group in groups:
                g=group['group'];case=sm['cases'][group['right_case']];amplitude=case['yaw_rate_target_rad_s'];speed=case['speed_mps'];temp=group['temperature_c'];normal=z[f'group_{g}_normal_prediction'];raw=z[f'group_{g}_normal_raw_context'];context_ref=[]
                for label,times in [('decision',x['decision_body_sample_time']),('views',x['view_body_sample_time'])]:
                    expected=body_expected(times,speed,amplitude,temp,onset,duration)
                    for key,value in zip(('body21','body21_valid','extension19','extension19_valid'),expected):
                        actual=z[f'group_{g}_{label}_{key}']
                        if value.dtype==bool:np.testing.assert_array_equal(actual,value)
                        else:maximum_body=max(maximum_body,float(abs(actual-value).max()));assert maximum_body<1e-12
                history=z[f'group_{g}_views_body21'][...,4].astype(np.float32)
                for branch,case_id in enumerate((group['right_case'],group['left_case'])):
                    scene=x['scene'][case_id].copy();scene[:plan['branch_frame']]=x['scene'][group['right_case'],:plan['branch_frame']];opponent=trace_opponents(scene,x['camera_receipt']+.005,mean,scale,model.temporal.fast_seconds,model.temporal.slow_seconds);context_ref.append(opponent)
                for arm in ('normal','context_off','context_reverse','context_delayed120'):
                    prediction=z[f'group_{g}_{arm}_prediction'];np.testing.assert_array_equal(prediction[:,:plan['branch_frame']+1],normal[:,:plan['branch_frame']+1]);np.testing.assert_array_equal(z[f'group_{g}_{arm}_body_ports'],z[f'group_{g}_normal_body_ports']);np.testing.assert_array_equal(z[f'group_{g}_{arm}_descending_ports'],z[f'group_{g}_normal_descending_ports']);np.testing.assert_array_equal(z[f'group_{g}_{arm}_raw_context'],raw);injected=z[f'group_{g}_{arm}_effective_context_yaw'];expected_yaw=history if arm in ('normal','context_off') else -history if arm=='context_reverse' else np.asarray([[profile(float(t)-100.-.12,amplitude,onset,duration)[0] for t in row] for row in x['view_body_sample_time']],dtype=np.float32);np.testing.assert_array_equal(injected[0],expected_yaw);np.testing.assert_array_equal(injected[1],expected_yaw)
                    for branch in range(2):
                        logit=(context_ref[branch].astype(float)*expected_yaw[...,None].astype(float)).reshape(len(history),256)@weight.T;expected=.5*np.tanh(logit)
                        if arm=='context_off':expected[:]=0
                        maximum_context=max(maximum_context,float(abs(expected-z[f'group_{g}_{arm}_effective_context'][branch]).max()));assert maximum_context<3e-6
                    mu=z[f'group_{g}_{arm}_muscles'].astype(float);logits=((mu-head.training_reference.numpy().astype(float))/head.training_scale.numpy().astype(float))@head.projection.weight.detach().numpy().astype(float).T+head.projection.bias.detach().numpy().astype(float);expected=np.concatenate([np.tanh(logits[...,:1]),1/(1+np.exp(-logits[...,1:]))],axis=-1);maximum_readout=max(maximum_readout,float(abs(expected-prediction).max()));assert maximum_readout<2e-6
                    delta=np.max(abs(prediction-normal),axis=(0,1));np.testing.assert_allclose(delta,group['branches'][arm]['post_prefix_max_abs_action_change_vs_normal'],atol=1e-12,rtol=0);margin=float(plan['expected_canonical_steering_sign_per_left_bend']*np.mean(prediction[1,plan['evaluation_first_frame']:,0]-prediction[0,plan['evaluation_first_frame']:,0])/2);assert margin==group['branches'][arm]['hypothesized_bend_margin']
                    if amplitude==0:zero_max=max(zero_max,float(abs(prediction-normal).max()));assert zero_max==0.
                effects.append(dict(group=g,target_yaw_rate=amplitude,temperature_c=temp,speed_mps=speed,context_off_axis_maximum_delta=group['branches']['context_off']['post_prefix_max_abs_action_change_vs_normal']))
        reports.append(dict(response=str(root),groups=len(groups),zero_rate_negative_control_max_difference=zero_max,effects=effects))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),stimuli_result_sha256=sha256(source/'result.json'),optimizer_updates=0,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_visual_body_rotation_audit',quad_trajectory_maximum_difference=maximum_pose,minimum_camera_bend_entry_margin_m=minimum_entry,raw_body_and_acceleration_maximum_difference=maximum_body,independent_held_trace_context_maximum_difference=maximum_context,independent_muscle_readout_maximum_difference=maximum_readout,same_prefix_first_branch_output_and_noncontext_ports_verified=True,known_and_unknown_sensor_masks_verified=True,reported_responses=reports,full_neural_replay_by_audit=False,actual_source_exposure_sync_verified=False,physiological_correctness_verified=False,actual_driving_improvement_verified=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent adaptive quadrature/analytic yaw and clocks, exact source sensor masks/values, held image-trace filtering and context projection, muscle readout and original/noncontext invariants. Full neural state evolution is source-bound, not rerun by this audit.'))

if __name__=='__main__':main()
