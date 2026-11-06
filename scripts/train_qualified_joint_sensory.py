"""One fixed TRAIN epoch of joint visual/native adapters at existing flight ports."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.joint_sensory_objective import joint_sensory_loss,project_sensory_weight
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read,detach_recurrence
from scripts.new_condition_joint_experiment import load_joint_experiment
from scripts.native_visual_experiment import tensor_state_equal
from scripts.train_qualified_sac_unified import context
from scripts.improve_sac_anticipation import command_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan)
    torch.set_num_threads(1);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    components,receipt,entries,items,extras,native,valid,baseline,pre=load_joint_experiment(plan);model,head,dn,optical,port,_=components
    protected=[(m,{k:v.clone() for k,v in m.state_dict().items()}) for m in (model,head,dn)];optical_before={k:v.clone() for k,v in optical.state_dict().items()};port_before={k:v.clone() for k,v in port.state_dict().items()}
    optical.projection.requires_grad_(True);port.projection.requires_grad_(True);parameters=[*optical.projection.parameters(),*port.projection.parameters()]
    if any(p.requires_grad for m in (model,head,dn) for p in m.parameters()):raise ValueError('Frozen model/head/DN unexpectedly trainable')
    optimizer=torch.optim.Adam(parameters,lr=plan['learning_rate']);run=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),candidate=receipt,entries=entries,seed=plan['seed'],train_episodes=[0,1],development_episode=2,trainable='Only existing learned optical and native body projection weights; normalized sensor/reference fields fixed',fixed='Original graph/model51 states, all muscle readout rows and DN',actor_qualified=False,vehicle_control_sent=False))
    for name in plan['source_sha256']:
        dest=run/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    holder=dict(e=0,i=0);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
    def descending(module,args,value):
        steer,power=value;return steer+dn(extras[holder['e']][holder['i']:holder['i']+1]),power
    def spatial(module,args,value):return value+optical(extras[holder['e']][holder['i']:holder['i']+1,64:])
    def body(module,args,value):
        e,i=holder['e'],holder['i'];expected=torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32)
        if not torch.equal(args[0],expected):raise ValueError('Native port joined to another body')
        return value+port(native[e][i:i+1],valid[e][i:i+1])
    handles=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(spatial),model.body_feedback.register_forward_hook(body)]
    updates=0;gradmax=0.;started=time.perf_counter();training=[];saved={};prior_gc=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    def rollout(e,learn=False,limit=None):
        nonlocal updates,gradmax
        d,c,_=items[e];holder['e']=e;scene=torch.tensor(d['scene'],dtype=torch.float32);labels=torch.tensor(d['labels'],dtype=torch.float32);anchor=torch.tensor(baseline[e],dtype=torch.float32);first,last=np.flatnonzero(d['fit_mask'])[[0,-1]];model.reset_episode();pred=[];mus=[];losses=[];supervised=0;total=len(scene) if limit is None else min(limit,len(scene))
        for i in range(total):
            holder['i']=i;learning=bool(learn and first<=i<=last)
            with torch.set_grad_enabled(learning):
                fn=model.sensory_training_decision if learning else model.sensory_decision
                _,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=goals,scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i));prediction=head(mu);pred.append(prediction[0].detach().numpy().copy());mus.append(mu[0].detach().numpy().copy())
                if learning and d['fit_mask'][i]:losses.append(joint_sensory_loss(prediction,labels[i],anchor[i],pre=bool(pre[e][i])));supervised+=1
                if learn and ((i+1)%plan['window']==0 or i==total-1):
                    if losses:
                        optimizer.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                        if not torch.isfinite(norm):raise ValueError('Nonfinite sensory gradient')
                        gradmax=max(gradmax,float(norm));optimizer.step();updates+=1;losses.clear();project_sensory_weight(optical.projection.weight,optical_before['projection.weight'],plan['maximum_weight_delta']);project_sensory_weight(port.projection.weight,port_before['projection.weight'],plan['maximum_weight_delta'])
                    detach_recurrence(model)
            if i%1000==0:write_json(run/'progress.json',dict(phase='training' if learn else 'fixed_replay',episode=e,event=i,total=total,optimizer_updates=updates,maximum_gradient_norm=gradmax,elapsed_seconds=time.perf_counter()-started))
        if learn:training.append(dict(episode=e,supervised_labels=supervised))
        return np.asarray(pred),np.asarray(mus)
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Declared CPU numerical mode unsupported')
        parity=[]
        for e in (0,1):
            prediction,_=rollout(e,limit=96);error=float(abs(prediction-baseline[e][:96]).max());parity.append(error)
            if error>2e-6:raise ValueError('Initial direct training path differs from audited source')
        for e in (0,1):rollout(e,learn=True)
        for module,before in protected:
            if not tensor_state_equal(module.state_dict(),before):raise ValueError('Frozen flight model/readout/DN changed')
        for module,before in [(optical,optical_before),(port,port_before)]:
            for key in before:
                if key!='projection.weight' and not torch.equal(module.state_dict()[key],before[key]):raise ValueError('Frozen sensory normalization changed')
        optical_delta=float(abs(optical.projection.weight.detach()-optical_before['projection.weight']).max());native_delta=float(abs(port.projection.weight.detach()-port_before['projection.weight']).max())
        if min(optical_delta,native_delta,gradmax)<=0:raise ValueError('Both sensory paths must actually learn')
        checkpoint=run/'joint-sensory.pt';torch.save(dict(schema='flybrain.qualified_joint_sensory.v1',parent_candidate_plan=plan['candidate_plan'],parent_candidate_plan_sha256=plan['candidate_plan_sha256'],optical=optical.state_dict(),body_port=port.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),optimizer_updates=updates,actor_qualified=False),checkpoint)
        write_json(run/'training-complete.json',dict(status='completed_training_pending_fixed_replay',optimizer_updates=updates,maximum_visual_weight_delta=optical_delta,maximum_native_weight_delta=native_delta,maximum_gradient_norm=gradmax,checkpoint_sha256=sha256(checkpoint)))
        comparisons=[]
        for e,(d,_,_) in enumerate(items):
            prediction,mus=rollout(e);baseline_all=command_metrics(baseline[e],d['labels'],d['fit_mask']);candidate_all=command_metrics(prediction,d['labels'],d['fit_mask']);baseline_pre=command_metrics(baseline[e],d['labels'],pre[e]);candidate_pre=command_metrics(prediction,d['labels'],pre[e]);comparisons.append(dict(episode=e,baseline=baseline_all,candidate=candidate_all,baseline_pre=baseline_pre,candidate_pre=candidate_pre,pre_labels=int(pre[e].sum())));saved.update({f'prediction_{e}':prediction,f'muscles_{e}':mus});write_json(run/'completed-episodes.json',dict(comparisons=comparisons))
        np.savez_compressed(run/'predictions.npz',**saved);dev=comparisons[2];gate=dev['candidate']['rmse'][0]<=.98*dev['baseline']['rmse'][0] and dev['candidate_pre']['rmse'][0]<=.98*dev['baseline_pre']['rmse'][0] and all(dev['candidate']['rmse'][k]<=1.02*dev['baseline']['rmse'][k] for k in (1,2))
        result.update(status='completed_qualified_joint_sensory_fit',optimizer_updates=updates,training_rows=training,initial_prefix_max_differences=parity,frozen_model_readout_dn_exact=True,normalizations_exact=True,maximum_visual_weight_delta=optical_delta,maximum_native_weight_delta=native_delta,maximum_gradient_norm=gradmax,comparisons=comparisons,development_two_percent_gate_passed=bool(gate),checkpoint_sha256=sha256(checkpoint),predictions_sha256=sha256(run/'predictions.npz'),termination_reason='One fixed TRAIN epoch at learned visual/body input projections, three full fixed continuous replays; consumed development only, no actor promotion or new corner/limit-driving qualification.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),optimizer_updates=updates)
    finally:
        for handle in handles:handle.remove()
        torch.set_flush_denormal(False);gc.enable() if prior_gc else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-started;write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
