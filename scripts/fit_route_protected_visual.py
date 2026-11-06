"""One joint sensory-boundary pass: synthetic function plus real TRAIN protection."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.functional_bend_objective import functional_bend_loss
from flybrain_drive.functional_bend_objective import functional_bend_loss
from flybrain_drive.functional_motion_overlay import exact_model
from flybrain_drive.graph import sha256
from flybrain_drive.joint_sensory_objective import project_sensory_weight
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.route_protected_visual_overlay import require_pooled_only_model
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.train_qualified_sac_unified import context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
    for path,expected in plan['source_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Sealed joint sensory source changed '+path)
    if sha256(plan['candidate_plan'])!=plan['candidate_plan_sha256'] or sha256(plan['reserved_test_plan'])!=plan['reserved_test_plan_sha256']:raise ValueError('Fixed parent or sealed test plan changed')
    components,motion,parent=load_readonly_motion_candidate(read(plan['candidate_plan']));model,head,dn,optical,body_port,_=components;oldfit=Path(read(plan['candidate_plan'])['motion']['fit']);om=read(oldfit/'manifest.json')['config'];_,_,entries,items,extras,native,valid,_,pre,features,maps,_,_,_=load_motion_experiment(om['plan'],'fine_motion')
    with np.load(oldfit/'predictions.npz') as z:real_anchor={e:z[f'prediction_{e}'].copy() for e in (0,1)}
    source=Path(plan['training_stimuli']);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];stimulus=sm['plan'];baseline=Path(plan['training_response']);br=read(baseline/'result.json');bm=read(baseline/'manifest.json')['config']
    if sha256(source/'result.json')!=plan['training_stimuli_result_sha256'] or sha256(source/'features.npz')!=sr['features_sha256'] or sha256(baseline/'result.json')!=plan['training_response_result_sha256'] or sha256(baseline/'responses.npz')!=br['responses_sha256'] or bm['candidate']!=parent:raise ValueError('Fixed TRAIN synthetic inputs/parent anchors required')
    with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
    with np.load(baseline/'responses.npz') as z:synthetic_anchor={k:z[k].copy() for k in z.files if k.endswith('_fine_prediction')}
    for module in components[:5]:module.eval().requires_grad_(False)
    protected=[{k:v.clone() for k,v in module.state_dict().items()} for module in components[:5]];motion_original={k:v.clone() for k,v in motion.state_dict().items()};real_tasks=[]
    for e in (0,1):
        d=items[e][0];blocks=(len(d['scene'])+plan['window']-1)//plan['window'];selected=np.union1d(np.unique(np.flatnonzero(pre[e])//plan['window']),np.linspace(0,blocks-1,plan['anchor_blocks_per_episode'],dtype=int))
        real_tasks.extend(dict(episode=e,begin=int(b*plan['window']),end=min(int((b+1)*plan['window']),len(d['scene']))) for b in selected)
    out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),parent_candidate=parent,entries=entries,train_episodes=[0,1],development_episode_used_for_fit=False,real_windows=real_tasks,seed=plan['seed'],actor_qualified=False,vehicle_control_sent=False,scope='Existing pooled sensory visual weight columns12+ and fine motion projection only. Chemical graph/readout/body/DN/static/normalization fixed. Parent-source numerical prefix states at explicit training cuts; final updated whole trajectories must be evaluated independently. No test features or teacher truth actor inputs.'))
    for path in plan['source_sha256']:
        target=out/'executed-source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    holder=dict(mode='real',episode=0,index=0,case=0);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);contract=sha256('configs/ac-virtual-navigation-body-v2.json');hooks=[];states={};updates=0;maximum_gradient=0.;records=[];before=gc.isenabled();gc.disable();start=time.perf_counter();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    def current_inputs():
        i=holder['index']
        if holder['mode']=='real':
            e=holder['episode'];j=int(maps[e][i]);return extras[e][i:i+1],features[e][j:j+1],native[e][i:i+1],valid[e][i:i+1]
        k=holder['case'];return torch.tensor(x['static'][k,i:i+1]),torch.tensor(x['motion'][k,i:i+1]),holder['native'],holder['native_valid']
    def descending(module,args,value):
        static,_,_,_=current_inputs();steer,power=value;return steer+dn(static),power
    def visual(module,args,value):
        static,fine,_,_=current_inputs();return value+optical(static[:,64:])+motion(fine)
    def body(module,args,value):
        _,_,sensed,mask=current_inputs();return value+body_port(sensed,mask)
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)]
    def step(i,learn):
        holder['index']=i
        if holder['mode']=='real':
            d,c,_=items[holder['episode']];scene=torch.tensor(d['scene'][i:i+1]);ctx=context(d,c,i);now=float(d['decision_time'][i])
        else:
            clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=holder['body'],holder['mask'];current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,values.copy(),mask.copy(),contract);history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,values.copy(),mask.copy(),contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now);scene=torch.tensor(x['scene'][holder['case'],i:i+1])
        with torch.set_grad_enabled(learn):
            fn=model.sensory_training_decision if learn else model.sensory_decision;_,muscles=fn(decision_time=now,racing_goals=goals,scene_features=scene,scene_valid=views,body_context=ctx);return head(muscles)[0]
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Fixed CPU numerical mode unavailable')
        write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(model));parity=[]
        for e in (0,1):
            holder.update(mode='real',episode=e);model.reset_episode();wanted={t['begin'] for t in real_tasks if t['episode']==e};maximum=0.
            for i in range(len(items[e][0]['scene'])):
                if i in wanted:states[(e,i)]=snapshot(model)
                prediction=step(i,False).numpy();maximum=max(maximum,float(abs(prediction-real_anchor[e][i]).max()))
                if maximum>2e-6:raise ValueError('Parent complete TRAIN trajectory differs')
                if i%1000==0:write_json(out/'progress.json',dict(phase='parent_full_prefix_collection',episode=e,event=i,total=len(items[e][0]['scene']),optimizer_updates=updates,maximum_parent_difference=maximum,elapsed_seconds=time.perf_counter()-start))
            parity.append(maximum)
        torch.save(states,out/'parent-prefix-states.pt');pooled_initial=model.visual.weight.detach().clone();motion_initial=motion.projection.weight.detach().clone();model.visual.weight.requires_grad_(True);motion.projection.weight.requires_grad_(True);parameters=[model.visual.weight,motion.projection.weight];optimizer=torch.optim.Adam(parameters,lr=plan['learning_rate']);rng.shuffle(real_tasks);synthetic_tasks=[dict(pair=k,temperature=t,group=3*(k//2)+j) for k in range(0,len(sm['cases']),2) for j,t in enumerate(stimulus['temperature_c'])];rng.shuffle(synthetic_tasks);slots=np.linspace(0,len(real_tasks)-1,len(synthetic_tasks),dtype=int);schedule=[];pending={int(slot):task for slot,task in zip(slots,synthetic_tasks)}
        for j,task in enumerate(real_tasks):
            schedule.append(('real',task))
            if j in pending:schedule.append(('synthetic',pending[j]))
        write_json(out/'training-schedule.json',schedule)
        for kind,task in schedule:
            if kind=='real':
                e=task['episode'];begin,end=task['begin'],task['end'];holder.update(mode='real',episode=e);restore(model,states[(e,begin)]);d=items[e][0];prediction=torch.stack([step(i,True) for i in range(begin,end)]);anchor=torch.tensor(real_anchor[e][begin:end]);loss=plan['real_anchor_weight']*((prediction-anchor).square()*prediction.new_tensor([4.,1.,1.])).sum(1).mean();mask=d['fit_mask'][begin:end]
                if mask.any():
                    weights=np.tile([4.,1.,1.],(end-begin,1));weights[pre[e][begin:end]]=[20.,1.,4.];target=torch.tensor(d['labels'][begin:end]);loss=loss+(((prediction-target).square()*torch.tensor(weights,dtype=torch.float32)).sum(1)[torch.tensor(mask)]).mean()
            else:
                k=task['pair'];case=sm['cases'][k];values,mask,sensed,validity=synthetic_body(case['speed_mps'],task['temperature'],body_port.reference.numpy());holder.update(mode='synthetic',case=k,body=values,mask=mask,native=torch.tensor(sensed[None],dtype=torch.float32),native_valid=torch.tensor(validity[None]));model.reset_episode()
                for i in range(stimulus['branch_frame']):step(i,False)
                prefix=snapshot(model);branches=[]
                for case_id in (k,k+1):
                    holder['case']=case_id;restore(model,prefix);sequence=[]
                    for i in range(stimulus['branch_frame'],stimulus['frame_count']):
                        learn=i>=stimulus['evaluation_first_frame'];command=step(i,learn)
                        if learn:sequence.append(command)
                    branches.append(torch.stack(sequence))
                anchor=torch.tensor(synthetic_anchor[f"group_{task['group']}_fine_prediction"][:,stimulus['evaluation_first_frame']:]);loss=functional_bend_loss(branches[0],branches[1],anchor[0],anchor[1],left_sign=stimulus['expected_canonical_steering_sign_per_left_bend'],target_margin=plan['target_margin'],direction_weight=plan['direction_weight'],anchor_weight=plan['synthetic_anchor_weight'])
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite joint sensory gradient')
            maximum_gradient=max(maximum_gradient,float(norm));optimizer.step();project_sensory_weight(model.visual.weight,pooled_initial,plan['maximum_pooled_weight_change']);project_sensory_weight(motion.projection.weight,motion_initial,plan['maximum_motion_weight_change'])
            with torch.no_grad():model.visual.weight[:,:12].copy_(pooled_initial[:,:12])
            updates+=1;records.append(dict(kind=kind,task=task,loss=float(loss.detach()),gradient_norm=float(norm)));write_json(out/'progress.json',dict(phase='joint_training',optimizer_updates=updates,total_updates=len(schedule),maximum_gradient_norm=maximum_gradient,elapsed_seconds=time.perf_counter()-start))
        require_pooled_only_model(model.state_dict(),protected[0],plan['maximum_pooled_weight_change'])
        if any(not exact_model(module.state_dict(),original) for module,original in zip(components[1:5],protected[1:])):raise ValueError('Frozen DN/static/native/readout changed')
        if any(not torch.equal(motion.state_dict()[k],v) for k,v in motion_original.items() if k!='projection.weight'):raise ValueError('Motion normalization changed')
        checkpoint=out/'route-protected.pt';torch.save(dict(schema='flybrain.route_protected_visual.v1',parent_candidate=parent,model=model.state_dict(),motion_port=motion.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),optical=optical.state_dict(),body_port=body_port.state_dict(),optimizer_updates=updates,actor_qualified=False),checkpoint);write_json(out/'training-records.json',records);result.update(status='completed_route_protected_visual_fit',optimizer_updates=updates,real_updates=sum(v['kind']=='real' for v in records),synthetic_updates=sum(v['kind']=='synthetic' for v in records),parent_full_train_max_differences=parity,maximum_gradient_norm=maximum_gradient,maximum_pooled_weight_change=float(abs(model.visual.weight-pooled_initial).max()),maximum_motion_weight_change=float(abs(motion.projection.weight-motion_initial).max()),checkpoint_sha256=sha256(checkpoint),training_records_sha256=sha256(out/'training-records.json'),parent_prefix_states_sha256=sha256(out/'parent-prefix-states.pt'),original_chemical_graph_body_readout_exact=True,reserved_test_features_read=False,termination_reason='One interleaved TRAIN-only sensory-boundary pass with original complete parent numerical prefixes at explicit cuts; independent continuous updated trajectory/test evaluation still required. No actor/vehicle authority.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),optimizer_updates=updates)
    finally:
        for hook in hooks:hook.remove()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
