"""One sealed TRAIN-texture pass at the existing fine-motion projection only."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.functional_bend_objective import functional_bend_loss
from flybrain_drive.functional_motion_overlay import exact_model
from flybrain_drive.graph import sha256
from flybrain_drive.joint_sensory_objective import project_sensory_weight
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.sac_steering_experiment import snapshot,restore


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    for name,expected in plan['source_sha256'].items():
        if sha256(name)!=expected:raise ValueError('Sealed functional fit source changed '+name)
    source=Path(plan['training_stimuli']);baseline=Path(plan['training_response']);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];br=read(baseline/'result.json');bm=read(baseline/'manifest.json')['config'];stimulus=sm['plan']
    if sr['status']!='completed_controlled_motion_factorial' or sha256(source/'result.json')!=plan['training_stimuli_result_sha256'] or sha256(source/'features.npz')!=sr['features_sha256'] or br['status']!='completed_controlled_motion_factorial_response' or sha256(baseline/'result.json')!=plan['training_response_result_sha256'] or bm['stimuli_result_sha256']!=sha256(source/'result.json') or sha256(baseline/'responses.npz')!=br['responses_sha256'] or sha256(plan['reserved_test_plan'])!=plan['reserved_test_plan_sha256']:raise ValueError('Exact TRAIN-only stimulus/anchor and sealed unused test plan required')
    if stimulus['texture_seed']==read(plan['reserved_test_plan'])['texture_seed']:raise ValueError('Reserved test texture equals TRAIN')
    components,motion,receipt=load_readonly_motion_candidate(read(stimulus['candidate_plan']));model,head,dn,optical,body_port,_=components
    if motion is None or receipt!=bm['candidate']:raise ValueError('Exact fixed motion parent required')
    for module in components[:5]:module.eval().requires_grad_(False)
    protected=[{k:v.clone() for k,v in module.state_dict().items()} for module in components[:5]];normalization={k:v.clone() for k,v in motion.state_dict().items() if k!='projection.weight'};initial=motion.projection.weight.detach().clone();motion.projection.weight.requires_grad_(True);optimizer=torch.optim.Adam([motion.projection.weight],lr=plan['learning_rate'])
    with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
    with np.load(baseline/'responses.npz') as z:anchors={key:z[key].copy() for key in z.files if key.endswith('_fine_prediction')}
    out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),parent_candidate=receipt,training_stimuli_result_sha256=sha256(source/'result.json'),reserved_test_plan_sha256=plan['reserved_test_plan_sha256'],seed=plan['seed'],actor_qualified=False,vehicle_control_sent=False,scope='Synthetic functional bend-order loss at existing motion projection. Existing model/readout/DN/static/native and normalization frozen. No test pixels/responses, actual driving labels or actor truth inputs.'))
    for name in plan['source_sha256']:
        dest=out/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    holder=dict(case=0,frame=0);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);contract=sha256('configs/ac-virtual-navigation-body-v2.json');updates=0;gradmax=0.;records=[];hooks=[];before=gc.isenabled();gc.disable();start=time.perf_counter();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    def descending(module,args,value):
        k,i=holder['case'],holder['frame'];steer,power=value;return steer+dn(torch.tensor(x['static'][k,i:i+1])),power
    def visual(module,args,value):
        k,i=holder['case'],holder['frame'];return value+optical(torch.tensor(x['static'][k,i:i+1,64:]))+motion(torch.tensor(x['motion'][k,i:i+1]))
    def body(module,args,value):return value+body_port(holder['native'],holder['native_valid'])
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)]
    def step(case,i,learn):
        holder.update(case=case,frame=i);clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=holder['body'],holder['mask'];current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,values.copy(),mask.copy(),contract);history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,values.copy(),mask.copy(),contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now)
        with torch.set_grad_enabled(learn):
            fn=model.sensory_training_decision if learn else model.sensory_decision;_,mu=fn(decision_time=now,racing_goals=goals,scene_features=torch.tensor(x['scene'][case,i:i+1]),scene_valid=views,body_context=ctx);return head(mu)[0]
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Fixed numerical mode unavailable')
        write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(model))
        for pair in range(0,len(sm['cases']),2):
            for temperature in stimulus['temperature_c']:
                case=sm['cases'][pair];values,mask,native,valid=synthetic_body(case['speed_mps'],temperature,body_port.reference.numpy());holder.update(body=values,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]));model.reset_episode()
                for i in range(stimulus['branch_frame']):step(pair,i,False)
                prefix=snapshot(model);branches=[]
                for k in (pair,pair+1):
                    restore(model,prefix);sequence=[]
                    for i in range(stimulus['branch_frame'],stimulus['frame_count']):
                        learn=i>=stimulus['evaluation_first_frame'];command=step(k,i,learn)
                        if learn:sequence.append(command)
                    branches.append(torch.stack(sequence))
                group=len(records);anchor=torch.tensor(anchors[f'group_{group}_fine_prediction'][:,stimulus['evaluation_first_frame']:]);loss=functional_bend_loss(branches[0],branches[1],anchor[0],anchor[1],left_sign=stimulus['expected_canonical_steering_sign_per_left_bend'],target_margin=plan['target_margin'],direction_weight=plan['direction_weight'],anchor_weight=plan['anchor_weight']);optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_([motion.projection.weight],1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite functional input gradient')
                gradmax=max(gradmax,float(norm));optimizer.step();project_sensory_weight(motion.projection.weight,initial,plan['maximum_weight_change']);updates+=1;records.append(dict(group=group,loss=float(loss.detach()),gradient_norm=float(norm),temperature_c=temperature,lateral_m=case['lateral_m'],speed_mps=case['speed_mps']));del branches,sequence,command,loss;write_json(out/'progress.json',dict(optimizer_updates=updates,total_updates=len(sm['cases'])//2*len(stimulus['temperature_c']),maximum_gradient_norm=gradmax,elapsed_seconds=time.perf_counter()-start))
        for module,original in zip(components[:5],protected):
            current=module.state_dict()
            if not exact_model(current,original):raise ValueError('Original frozen component changed')
        if any(not torch.equal(motion.state_dict()[k],v) for k,v in normalization.items()):raise ValueError('Motion normalization changed')
        delta=float((motion.projection.weight.detach()-initial).abs().max())
        if updates!=18 or gradmax<=0 or delta<=0:raise ValueError('Expected one complete functional pass with real updates required')
        checkpoint=out/'functional-motion.pt';torch.save(dict(schema='flybrain.functional_motion_port.v1',parent_candidate=receipt,motion_port=motion.state_dict(),model=model.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),optical=optical.state_dict(),body_port=body_port.state_dict(),maximum_weight_change=plan['maximum_weight_change'],optimizer_updates=updates,actor_qualified=False),checkpoint);write_json(out/'training-records.json',records);result.update(status='completed_controlled_motion_function_fit',optimizer_updates=updates,maximum_gradient_norm=gradmax,maximum_weight_change=delta,original_components_exact=True,motion_normalization_exact=True,reserved_test_read_by_optimizer=False,checkpoint_sha256=sha256(checkpoint),training_records_sha256=sha256(out/'training-records.json'),termination_reason='One TRAIN synthetic-texture pass of bend-order/three-axis anchor at existing fine-motion projection; immutable graph/readout/body preserved, test unused and no actual driving qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),optimizer_updates=updates)
    finally:
        for hook in hooks:hook.remove()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
