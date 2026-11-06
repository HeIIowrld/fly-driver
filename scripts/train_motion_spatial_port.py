"""Matched fine-motion vs duplicated existing-motion input-port learning only."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.joint_sensory_objective import joint_sensory_loss,project_sensory_weight
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read,detach_recurrence
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.native_visual_experiment import tensor_state_equal
from scripts.train_qualified_sac_unified import context
from scripts.improve_sac_anticipation import command_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--arm',choices=['fine_motion','coarse_duplicate'],required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    (components,receipt,entries,items,extras,native,valid,baseline,pre,features,maps,coarse,reference,scale)=load_motion_experiment(plan,a.arm)
    model,head,dn,optical,body_port,_=components;protected=[(module,{k:v.clone() for k,v in module.state_dict().items()}) for module in components[:5]]
    motion=MotionSpatialResidualPort(reference,scale,len(model.motor.backbone.vision_ports),maximum_logit_delta=plan['maximum_logit_delta']);normalization={k:v.clone() for k,v in motion.state_dict().items() if k!='projection.weight'};optimizer=torch.optim.Adam(motion.parameters(),lr=plan['learning_rate'])
    run=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),candidate=receipt,entries=entries,seed=plan['seed'],train_episodes=[0,1],development_episode=2,actor_qualified=False,vehicle_control_sent=False,trainable_parameters=sum(p.numel() for p in motion.parameters()),trainable='New zero-initialized current-camera motion residual projection only',fixed='Original graph/model51 states, DN/static optical/native body/all readout rows and normalizations',scope='Matched TRAIN epoch; coarse control duplicates only old actor information. New TRAIN/consumed Monza development, no unseen-corner or student control qualification.'))
    for name in plan['source_sha256']:
        dest=run/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    holder=dict(e=0,i=0);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
    def descending(module,args,value):
        e,i=holder['e'],holder['i'];steer,power=value;return steer+dn(extras[e][i:i+1]),power
    def visual(module,args,value):
        e,i=holder['e'],holder['i'];j=int(maps[e][i]);return value+optical(extras[e][i:i+1,64:])+motion(features[e][j:j+1])
    def body(module,args,value):
        e,i=holder['e'],holder['i']
        if not torch.equal(args[0],torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32)):raise ValueError('Body/native receiver differs')
        return value+body_port(native[e][i:i+1],valid[e][i:i+1])
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)];updates=0;gradmax=0.;start=time.perf_counter();training=[];saved={};before_gc=gc.isenabled();gc.disable();out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    def rollout(e,learn=False,limit=None):
        nonlocal updates,gradmax
        d,c,_=items[e];holder['e']=e;scene=torch.tensor(d['scene'],dtype=torch.float32);target=torch.tensor(d['labels'],dtype=torch.float32);anchor=torch.tensor(baseline[e],dtype=torch.float32);first,last=np.flatnonzero(d['fit_mask'])[[0,-1]];model.reset_episode();pred=[];mus=[];losses=[];labels=0;total=len(scene) if limit is None else min(limit,len(scene))
        for i in range(total):
            holder['i']=i;learning=bool(learn and first<=i<=last)
            with torch.set_grad_enabled(learning):
                fn=model.sensory_training_decision if learning else model.sensory_decision
                _,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=goals,scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i));command=head(mu);pred.append(command[0].detach().numpy().copy());mus.append(mu[0].detach().numpy().copy())
                if learning and d['fit_mask'][i]:losses.append(joint_sensory_loss(command,target[i],anchor[i],pre=bool(pre[e][i])));labels+=1
                if learn and ((i+1)%plan['window']==0 or i==total-1):
                    if losses:
                        optimizer.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();norm=torch.nn.utils.clip_grad_norm_(motion.parameters(),1.)
                        if not torch.isfinite(norm):raise ValueError('Nonfinite motion gradient')
                        gradmax=max(gradmax,float(norm));optimizer.step();updates+=1;losses.clear();project_sensory_weight(motion.projection.weight,torch.zeros_like(motion.projection.weight),plan['maximum_weight_delta'])
                    detach_recurrence(model)
            if i%1000==0:write_json(run/'progress.json',dict(arm=a.arm,phase='training' if learn else 'fixed_replay',episode=e,event=i,total=total,optimizer_updates=updates,maximum_gradient_norm=gradmax,elapsed_seconds=time.perf_counter()-start))
        if learn:training.append(dict(episode=e,supervised_labels=labels))
        return np.asarray(pred),np.asarray(mus)
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Declared CPU numerical mode unsupported')
        parity=[]
        for e in (0,1):
            pred,_=rollout(e,limit=96);error=float(abs(pred-baseline[e][:96]).max());parity.append(error)
            if error>2e-6:raise ValueError('Zero motion port changed original candidate')
        for e in (0,1):rollout(e,learn=True)
        for module,state in protected:
            if not tensor_state_equal(module.state_dict(),state):raise ValueError('Frozen original component changed')
        for key,value in normalization.items():
            if not torch.equal(motion.state_dict()[key],value):raise ValueError('TRAIN-only motion normalization changed')
        delta=float(motion.projection.weight.detach().abs().max())
        if min(delta,gradmax)<=0:raise ValueError('No new motion learning occurred')
        checkpoint=run/'motion-spatial.pt';torch.save(dict(schema='flybrain.motion_spatial_port.v1',arm=a.arm,parent_candidate=receipt,motion_port=motion.state_dict(),maximum_logit_delta=motion.maximum_logit_delta,readout=head.state_dict(),adapter=dn.state_dict(),optical=optical.state_dict(),body_port=body_port.state_dict(),optimizer_updates=updates,actor_qualified=False),checkpoint)
        write_json(run/'training-complete.json',dict(status='completed_training_pending_fixed_replay',optimizer_updates=updates,maximum_weight_delta=delta,maximum_gradient_norm=gradmax,checkpoint_sha256=sha256(checkpoint)))
        reports=[]
        for e,(d,c,_) in enumerate(items):
            pred,mu=rollout(e);reports.append(dict(episode=e,baseline=command_metrics(baseline[e],d['labels'],d['fit_mask']),candidate=command_metrics(pred,d['labels'],d['fit_mask']),baseline_pre=command_metrics(baseline[e],d['labels'],pre[e]),candidate_pre=command_metrics(pred,d['labels'],pre[e]),pre_labels=int(pre[e].sum())));saved.update({f'prediction_{e}':pred,f'muscles_{e}':mu});write_json(run/'completed-episodes.json',dict(comparisons=reports))
        np.savez_compressed(run/'predictions.npz',**saved);dev=reports[2];gate=dev['candidate']['rmse'][0]<=.98*dev['baseline']['rmse'][0] and dev['candidate_pre']['rmse'][0]<=.98*dev['baseline_pre']['rmse'][0] and all(dev['candidate']['rmse'][k]<=1.02*dev['baseline']['rmse'][k] for k in (1,2))
        out.update(status='completed_motion_spatial_port_fit',arm=a.arm,optimizer_updates=updates,training_rows=training,initial_zero_port_prefix_differences=parity,original_components_exact=True,motion_normalization_exact=True,maximum_weight_delta=delta,maximum_gradient_norm=gradmax,comparisons=reports,development_two_percent_gate_passed=bool(gate),checkpoint_sha256=sha256(checkpoint),predictions_sha256=sha256(run/'predictions.npz'),termination_reason='One matched TRAIN epoch at new visual input projection; all old visual/body/flight/readout components frozen. No actor promotion, live integration or student limit-driving qualification.')
    except Exception as error:out.update(error=type(error).__name__,reason=str(error),optimizer_updates=updates)
    finally:
        for handle in hooks:handle.remove()
        torch.set_flush_denormal(False);gc.enable() if before_gc else gc.disable()
    out['elapsed_seconds']=time.perf_counter()-start;write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
