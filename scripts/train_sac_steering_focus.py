"""Steering-focused matched continuation through the fixed flight circuit.

Only engineering sensory ports and row0 of the six-muscle vehicle readout may
learn. Pedal readout rows, measured graph and all51 original parameters/buffers
are exact. Training data remain two teacher laps, development already consumed.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.anticipation_targets import ridge_fit,ridge_predict
from flybrain_drive.sac_unified_student import detach_recurrence,read
from scripts.improve_sac_anticipation import command_metrics
from scripts.train_qualified_sac_unified import context
from scripts.sac_steering_experiment import dataset,assembly,onsets,SOURCE,SOURCE_SHA


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['control','descending','optical'],required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(780502);np.random.seed(780502)
 plan=read(a.plan)
 if plan['program_sha256']!=sha256(__file__) or plan['helper_sha256']!=sha256('scripts/sac_steering_experiment.py') or plan['source_checkpoint_sha256']!=SOURCE_SHA:raise ValueError('Sealed plan/source differs')
 entries,items,extras=dataset();model,readout,dn,base,lineage=assembly()
 original_readout={k:v.clone() for k,v in readout.state_dict().items()}
 raw=np.concatenate([x[d['fit_mask'],64:] for x,(d,c,e) in zip(extras[:2],items[:2])])
 optical=RetinalResidualPort(torch.tensor(raw.mean(0)),torch.tensor(raw.std(0)).clamp_min(1e-4),len(model.motor.backbone.vision_ports))
 dn.requires_grad_(a.arm=='descending');optical.requires_grad_(a.arm=='optical');readout.requires_grad_(True)
 # Row-level gradient mask; all old body/visual parameters remain requires_grad=False.
 handles=[readout.projection.weight.register_hook(lambda g:torch.cat([g[:1],torch.zeros_like(g[1:])])),
          readout.projection.bias.register_hook(lambda g:torch.cat([g[:1],torch.zeros_like(g[1:])]))]
 parameters=list(readout.parameters())+([dn.projection.weight] if a.arm=='descending' else list(optical.parameters()) if a.arm=='optical' else [])
 groups=[dict(params=list(readout.parameters()),lr=.0003)]
 if a.arm!='control':groups.append(dict(params=parameters[len(list(readout.parameters())):],lr=.0003,weight_decay=.001))
 optimizer=torch.optim.Adam(groups)
 run=begin_run(a.output,dict(arguments=vars(a),seed=780502,program_sha256=sha256(__file__),helper_sha256=sha256('scripts/sac_steering_experiment.py'),plan_sha256=sha256(a.plan),source_checkpoint_sha256=SOURCE_SHA,entries=entries,lineage=lineage,
    hypothesis='Steering-focused existing DN contrast or zero-initialized residual static input at original visual42 port; not new measured connections',
    all_old_model_state_entries_frozen=51,pedal_readout_rows_frozen=True,source_adapter_frozen=a.arm!='descending',
    epochs=1,window=32,loss_weights=[16,1,1],pre_yaw_transition_steering_multiplier=4,
    final_readout_refit='TRAIN-only fixed ridge .001 on six muscle atanh steering; pedal rows unchanged',actor_deployed=False,vehicle_control_sent=False))
 holder={'extra':None};goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
 def dn_hook(m,inputs,value):
  steering,power=value;return steering+dn(holder['extra']),power
 def optical_hook(m,inputs,value):return value+optical(holder['extra'][:,64:])
 h=model.motor.command.register_forward_hook(dn_hook);oh=model.visual.register_forward_hook(optical_hook) if a.arm=='optical' else None
 updates=0;start=time.monotonic();out=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 def rollout(index,learn=False,stop=None):
  nonlocal updates
  d,c,_=items[index];scene=torch.tensor(d['scene'],dtype=torch.float32);extra=torch.tensor(extras[index],dtype=torch.float32);labels=torch.tensor(d['labels'],dtype=torch.float32)
  pre=np.zeros(len(scene),dtype=bool)
  for w in onsets(d):pre|=(d['decision_time']>=w['time']-.8)&(d['decision_time']<=w['time']-.08)&(abs(d['body'][:,4])<.07)
  first,last=np.flatnonzero(d['fit_mask'])[[0,-1]];model.reset_episode();preds=[];muscles=[];losses=[]
  for i in range(len(scene) if stop is None else stop):
   learning=bool(learn and first<=i<=last);holder['extra']=extra[i:i+1]
   with torch.set_grad_enabled(learning):
    fn=model.sensory_training_decision if learning else model.sensory_decision
    _,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=goals,scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i))
    pr=readout(mu);preds.append(pr[0].detach().numpy().copy());muscles.append(mu[0].detach().numpy().copy())
    if learning and d['fit_mask'][i]:
     weights=pr.new_tensor([64. if pre[i] else 16.,1.,1.]);losses.append(((pr[0]-labels[i]).square()*weights).mean())
    if learn and ((i+1)%32==0 or i==len(scene)-1):
     if losses:
      optimizer.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(parameters,1.)
      if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in parameters):raise ValueError('Nonfinite gradient')
      optimizer.step();updates+=1;losses.clear()
     detach_recurrence(model)
   if i%1000==0:write_json(run/'progress.json',dict(arm=a.arm,episode=index,event=i,phase='training' if learn else 'fixed_evaluation',optimizer_updates=updates,elapsed_seconds=time.monotonic()-start))
  return np.asarray(preds),np.asarray(muscles)
 try:
  initial,_=rollout(2,stop=64)
  with np.load(SOURCE/'predictions.npz') as z:error=float(abs(initial-z['dev'][:64]).max())
  if error>2e-6:raise ValueError('All arms must start exact source-neutral')
  if a.arm!='control':
   for i in [0,1]:rollout(i,True)
  saved={};fixed=[]
  for i in [0,1]:
   pr,mu=rollout(i);saved[f'train_{i}_muscles']=mu;fixed.append(mu[items[i][0]['fit_mask']])
  x=np.concatenate(fixed);y=np.concatenate([d['labels'][d['fit_mask'],:1] for d,c,e in items[:2]])
  def normalized(mu):
   return ((torch.tensor(mu)-readout.training_reference)/readout.training_scale).numpy()
  fit=ridge_fit(normalized(x),np.arctanh(y),.001)
  # Exact algebra into the unchanged six-muscle normalized linear projection.
  beta=fit['weights'][:,0]/fit['scale'];bias=fit['target_mean'][0]-fit['mean']@beta
  with torch.no_grad():
   readout.projection.weight[0]=torch.tensor(beta,dtype=torch.float32)
   readout.projection.bias[0]=float(bias)
  train=[]
  for i in [0,1]:
   mu=saved[f'train_{i}_muscles'];pr=readout(torch.tensor(mu)).detach().numpy();saved[f'train_{i}']=pr
   np.testing.assert_allclose(pr[:,:1],np.tanh(ridge_predict(fit,normalized(mu))),atol=2e-6,rtol=1e-5)
   train.append(command_metrics(pr,items[i][0]['labels'],items[i][0]['fit_mask']))
  pred,mu=rollout(2);saved['dev']=pred;saved['dev_muscles']=mu
  for k,v in model.state_dict().items():
   b=base['model'][k]
   if v.layout!=torch.strided:
    x1,x2=v.to_sparse_coo().coalesce(),b.to_sparse_coo().coalesce();same=torch.equal(x1.indices(),x2.indices()) and torch.equal(x1.values(),x2.values())
   else:same=torch.equal(v,b)
   if not same:raise ValueError('Protected original state changed: '+k)
  for k in ['training_reference','training_scale']:
   if not torch.equal(readout.state_dict()[k],original_readout[k]):raise ValueError('Calibration changed')
  for k in ['projection.weight','projection.bias']:
   if not torch.equal(readout.state_dict()[k][1:],original_readout[k][1:]):raise ValueError('Pedal readout changed')
  file=run/'steering.pt';torch.save(dict(schema='flybrain.steering_focused.v1',source_checkpoint_sha256=SOURCE_SHA,baseline_checkpoint_sha256=base.get('baseline_checkpoint_sha256'),arm=a.arm,adapter=dn.state_dict(),optical=optical.state_dict() if a.arm=='optical' else None,readout=readout.state_dict(),actor_qualified=False,optimizer_updates=updates),file)
  np.savez_compressed(run/'predictions.npz',**saved)
  out.update(status='completed_fixed_flight_steering_focus',optimizer_updates=updates,train=train,dev=command_metrics(pred,items[2][0]['labels'],items[2][0]['fit_mask']),checkpoint_sha256=sha256(file),predictions_sha256=sha256(run/'predictions.npz'),initial_source_prefix_max_error=error,old_model_state_entries_exact=51,pedal_readout_rows_exact=True,termination_reason='Fixed epoch and TRAIN-only muscle readout refit, complete fixed train/dev replay finished')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),optimizer_updates=updates,termination_reason='Explicit steering fit failure')
 finally:
  h.remove()
  if oh is not None:oh.remove()
  for hh in handles:hh.remove()
 out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
