"""Matched current/future supervision of the native flight visual port only.

No actuator, simulator, future-observation or actor-input change. Every actor
command still comes from the same six muscles. Auxiliary forecast is training
and diagnostic only and never supplies a vehicle command.
"""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.bounded_steering_fit import fit_bounded_steering
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read,detach_recurrence
from flybrain_drive.visual_anticipatory_objective import visual_anticipatory_loss
from scripts.native_visual_experiment import load_experiment,tensor_state_equal
from scripts.train_qualified_sac_unified import context
from scripts.probe_protected_anticipatory_steering import future_label_join
from scripts.fit_bounded_sac_steering import metrics
from scripts.improve_sac_anticipation import command_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--arm',required=True,choices=['present','future200'])
    a=parser.parse_args();plan=read(a.plan)
    for path,expected in plan['source_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Sealed experiment source changed: '+path)
    if sha256(plan['reference_plan'])!=plan['reference_plan_sha256']:raise ValueError('Reference plan changed')
    torch.set_num_threads(1);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    components,receipt,entries,items,extras,native,valid,pre,baseline,muscles=load_experiment(plan['reference_plan'])
    model,head,dn,optical,port,_=components
    protected=[(m,{k:v.clone() for k,v in m.state_dict().items()}) for m in (model,head,dn,port)]
    optical_before={k:v.clone() for k,v in optical.state_dict().items()}
    optical.requires_grad_(True)
    if any(p.requires_grad for m in (model,head,dn,port) for p in m.parameters()):raise ValueError('Frozen component unexpectedly trainable')
    joins=[future_label_join(d['decision_time'],d['fit_mask']&d['label_valid'],.2,.06) for d,_,_ in items]
    xs=[];ys=[]
    for e in (0,1):
        d=items[e][0];j,ok=joins[e];ids=np.flatnonzero(pre[e]&ok)
        x=((torch.tensor(muscles[e])-head.training_reference)/head.training_scale).numpy()
        xs.append(x[ids]);ys.append(d['labels'][j[ids],0])
    x=np.concatenate(xs);y=np.concatenate(ys)
    # Same TRAIN-only frozen forecast in both arms, even in the no-aux control.
    auxiliary=fit_bounded_steering(x,y,np.ones(len(y)),head.projection.weight[0].numpy(),float(head.projection.bias[0]),.001)
    aux_beta=torch.tensor(auxiliary['beta'],dtype=torch.float32);aux_bias=torch.tensor(auxiliary['bias'],dtype=torch.float32)
    optimizer=torch.optim.Adam(optical.parameters(),lr=plan['learning_rate'])
    run=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),entries=entries,
        native_receipt=receipt,seed=plan['seed'],train_episodes=[0,1],development_episode=2,
        actor_qualified=False,vehicle_control_sent=False,trainable='Only inherited retinal256 -> existing visual-port projection',
        fixed='Original flight graph/model, DN, native/body paths, all six-muscle readout rows/calibration',
        development_scope='Consumed repeated Monza; no new-corner, temperature generalization or student driving qualification'))
    for name in plan['source_sha256']:
        dest=run/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    write_json(run/'auxiliary-fit.json',{k:v for k,v in auxiliary.items() if k not in ('beta','mean','scale')})
    holder=dict(e=0,i=0,extra=None);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
    def descending(module,args,value):
        steer,power=value;return steer+dn(holder['extra']),power
    def spatial(module,args,value):return value+optical(holder['extra'][:,64:])
    def body(module,args,value):
        e,i=holder['e'],holder['i']
        if not torch.equal(args[0],torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32)):raise ValueError('Native receiver changed')
        return value+port(native[e][i:i+1],valid[e][i:i+1])
    handles=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(spatial),model.body_feedback.register_forward_hook(body)]
    updates=0;gradient_max=0.;start=time.monotonic();training_rows=[];saved={}
    out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    def forecast(mu):return torch.tanh(((mu-head.training_reference)/head.training_scale)@aux_beta+aux_bias)
    def rollout(e,learn=False,limit=None):
        nonlocal updates,gradient_max
        d,c,_=items[e];holder['e']=e;scene=torch.tensor(d['scene'],dtype=torch.float32)
        labels=torch.tensor(d['labels'],dtype=torch.float32);anchor=torch.tensor(baseline[e],dtype=torch.float32)
        first,last=np.flatnonzero(d['fit_mask'])[[0,-1]];model.reset_episode();pred=[];mus=[];forecasts=[];losses=[];count=0;future_count=0
        total=len(scene) if limit is None else min(limit,len(scene))
        for i in range(total):
            learning=bool(learn and first<=i<=last);holder.update(i=i,extra=extras[e][i:i+1])
            with torch.set_grad_enabled(learning):
                fn=model.sensory_training_decision if learning else model.sensory_decision
                _,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=goals,
                    scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i))
                command=head(mu);future=forecast(mu)
                pred.append(command[0].detach().numpy().copy());mus.append(mu[0].detach().numpy().copy());forecasts.append(float(future.detach()[0]))
                if learning and d['fit_mask'][i]:
                    j,ok=joins[e];active=bool(a.arm=='future200' and pre[e][i] and ok[i])
                    losses.append(visual_anticipatory_loss(command,labels[i],anchor[i,1:],pre=bool(pre[e][i]),
                        forecast=future if active else None,future_target=labels[j[i],0] if active else None))
                    count+=1;future_count+=int(active)
                if learn and ((i+1)%plan['window']==0 or i==total-1):
                    if losses:
                        optimizer.zero_grad(set_to_none=True);torch.stack(losses).mean().backward()
                        norm=torch.nn.utils.clip_grad_norm_(optical.parameters(),1.)
                        if not torch.isfinite(norm):raise ValueError('Nonfinite visual gradient')
                        gradient_max=max(gradient_max,float(norm));optimizer.step();updates+=1;losses.clear()
                        with torch.no_grad():
                            initial=optical_before['projection.weight'];delta=(optical.projection.weight-initial).clamp(-plan['maximum_weight_delta'],plan['maximum_weight_delta'])
                            optical.projection.weight.copy_(initial+delta)
                    detach_recurrence(model)
            if i%500==0:
                write_json(run/'progress.json',dict(arm=a.arm,phase='training' if learn else 'fixed_evaluation',episode=e,event=i,total=total,
                    optimizer_updates=updates,maximum_gradient_norm=gradient_max,elapsed_seconds=time.monotonic()-start))
        if learn:training_rows.append(dict(episode=e,current_rows=count,auxiliary_future_rows=future_count))
        return np.asarray(pred),np.asarray(mus),np.asarray(forecasts)
    def checkpoint():
        return dict(schema='flybrain.native_visual_anticipatory.v1',reference_plan=plan['reference_plan'],reference_plan_sha256=plan['reference_plan_sha256'],
            arm=a.arm,optical=optical.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),body_port=port.state_dict(),
            auxiliary_beta=aux_beta,auxiliary_bias=aux_bias,auxiliary_is_actor_output=False,optimizer_updates=updates,actor_qualified=False)
    try:
        initial,_,_=rollout(0,limit=96)
        initial_error=float(np.max(abs(initial-baseline[0][:len(initial)])))
        if initial_error>2e-6:raise ValueError('Initial native visual candidate differs from reference')
        for e in (0,1):rollout(e,learn=True)
        for module,before in protected:
            if not tensor_state_equal(module.state_dict(),before):raise ValueError('Frozen component changed')
        for key in ('reference','scale'):
            if not torch.equal(optical.state_dict()[key],optical_before[key]):raise ValueError('TRAIN visual normalization changed')
        change=float(torch.max(abs(optical.projection.weight.detach()-optical_before['projection.weight'])))
        if gradient_max<=0 or change<=0:raise ValueError('No visual learning occurred')
        file=run/'visual-anticipation.pt';torch.save(checkpoint(),file)
        write_json(run/'training-complete.json',dict(status='completed_training_pending_fixed_evaluation',optimizer_updates=updates,training_rows=training_rows,
            maximum_visual_gradient_norm=gradient_max,maximum_visual_weight_delta=change,checkpoint_sha256=sha256(file),actor_qualified=False))
        reports=[]
        for e,(d,_,_) in enumerate(items):
            pred,mu,fc=rollout(e);j,ok=joins[e];mask=pre[e]&ok
            reports.append(dict(episode=e,baseline=metrics(baseline[e][:,0],d,pre[e]),candidate=metrics(pred[:,0],d,pre[e]),
                baseline_axes=command_metrics(baseline[e],d['labels'],d['fit_mask']),candidate_axes=command_metrics(pred,d['labels'],d['fit_mask']),
                auxiliary_future_pre_rmse=float(np.sqrt(np.mean((fc[mask]-d['labels'][j[mask],0])**2))),future_pre_count=int(mask.sum())))
            saved.update({f'prediction_{e}':pred,f'muscles_{e}':mu,f'forecast_{e}':fc,f'future_indices_{e}':j,f'future_valid_{e}':ok})
            write_json(run/'completed-episodes.json',dict(comparisons=reports))
        np.savez_compressed(run/'predictions.npz',**saved)
        dev=reports[2]
        gate=all(dev['candidate'][k]<=.98*dev['baseline'][k] for k in ('all_rmse','pre_rmse')) and all(dev['candidate_axes']['rmse'][k]<=1.02*dev['baseline_axes']['rmse'][k] for k in (1,2))
        out.update(status='completed_native_visual_anticipation_fit',comparisons=reports,optimizer_updates=updates,training_rows=training_rows,
            initial_reference_max_error=initial_error,frozen_components_exact=True,maximum_visual_weight_delta=change,
            maximum_visual_gradient_norm=gradient_max,development_two_percent_gate_passed=gate,
            checkpoint_sha256=sha256(file),predictions_sha256=sha256(run/'predictions.npz'),
            termination_reason='One fixed TRAIN epoch and three complete fixed continuous replays; no actor promotion')
    except Exception as error:
        out.update(error=type(error).__name__,reason=str(error),optimizer_updates=updates,termination_reason='Explicit native visual learning/evidence failure; no replacement fit')
    finally:
        for handle in handles:handle.remove()
    out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
