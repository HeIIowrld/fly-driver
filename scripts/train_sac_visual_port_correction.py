"""Steering-specific independent visual corrections at four existing DN ports.

Frozen sensory graph, source DN/optical adapters and every muscle-readout row.
Only a zero-neutral current-camera320->four logit correction is learned.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.visual_port_current_correction import VisualPortCurrentCorrection
from flybrain_drive.sac_unified_student import detach_recurrence,read
from scripts.sac_steering_experiment import dataset,assembly,onsets,SOURCE_SHA
from scripts.train_qualified_sac_unified import context
from scripts.improve_sac_anticipation import command_metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(780506);np.random.seed(780506)
 plan=read(a.plan);source=Path(a.source);sr=read(source/'result.json')
 if plan['program_sha256']!=sha256(__file__) or plan['correction_sha256']!=sha256('src/flybrain_drive/visual_port_current_correction.py') or sr['checkpoint_sha256']!=plan['source_checkpoint_sha256']:raise ValueError('Sealed source/experiment differs')
 entries,items,extras=dataset();model,head,dn,base,lineage=assembly(a.source)
 cp=torch.load(source/'steering.pt',map_location='cpu',weights_only=True);ac=cp['optical']
 optical=RetinalResidualPort(ac['reference'],ac['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(ac)
 for module in [model,head,dn,optical]:module.requires_grad_(False)
 correction=VisualPortCurrentCorrection(dn.reference,dn.scale)
 optimizer=torch.optim.Adam(correction.parameters(),lr=.001,weight_decay=.001)
 with np.load(source/'predictions.npz') as z:source_predictions=[z[f'train_{i}'].copy() for i in [0,1]];source_dev=z['dev'].copy()
 run=begin_run(a.output,dict(arguments=vars(a),seed=780506,program_sha256=sha256(__file__),plan_sha256=sha256(a.plan),source_checkpoint_sha256=sr['checkpoint_sha256'],ancestor_checkpoint_sha256=SOURCE_SHA,entries=entries,lineage=lineage,
     feature='Current original static64 plus discarded static256 only',target_port_ids=model.motor.backbone.spec['steering_input_ids'],hypothesis='Independent engineering corrections to four existing flight DN currents; not new measured wiring',epochs=1,window=64,learning_rate=.001,steering_weight=16,preweight=4,pedal_source_anchor_weight=64,
     fixed='All51 original model entries, source DN/optical adapters, all three muscle-readout rows and calibration',actor_deployed=False,vehicle_control_sent=False))
 holder={'extra':None};goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
 def hook(m,args,value):
  steering,power=value;return steering+correction(holder['extra'],dn(holder['extra'])),power
 def optical_hook(m,args,value):return value+optical(holder['extra'][:,64:])
 h=model.motor.command.register_forward_hook(hook);oh=model.visual.register_forward_hook(optical_hook)
 updates=0;start=time.monotonic();out=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 def rollout(index,learn=False,stop=None):
  nonlocal updates
  d,c,_=items[index];scene=torch.tensor(d['scene'],dtype=torch.float32);extra=torch.tensor(extras[index],dtype=torch.float32);target=torch.tensor(d['labels'],dtype=torch.float32)
  anchor=torch.tensor(source_predictions[index],dtype=torch.float32) if index<2 else None
  pre=np.zeros(len(scene),bool)
  for w in onsets(d):pre|=(d['decision_time']>=w['time']-.8)&(d['decision_time']<=w['time']-.08)&(abs(d['body'][:,4])<.07)
  first,last=np.flatnonzero(d['fit_mask'])[[0,-1]];model.reset_episode();preds=[];muscles=[];losses=[]
  for i in range(len(scene) if stop is None else stop):
   learning=bool(learn and first<=i<=last);holder['extra']=extra[i:i+1]
   with torch.set_grad_enabled(learning):
    fn=model.sensory_training_decision if learning else model.sensory_decision
    _,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=goals,scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i));pr=head(mu)
    preds.append(pr[0].detach().numpy().copy());muscles.append(mu[0].detach().numpy().copy())
    if learning and d['fit_mask'][i]:losses.append((64. if pre[i] else 16.)*(pr[0,0]-target[i,0]).square()+64*(pr[0,1:]-anchor[i,1:]).square().mean())
    if learn and ((i+1)%64==0 or i==len(scene)-1):
     if losses:
      optimizer.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(correction.parameters(),1.)
      if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in correction.parameters()):raise ValueError('Nonfinite gradient')
      optimizer.step();updates+=1;losses.clear()
     detach_recurrence(model)
   if i%1000==0:write_json(run/'progress.json',dict(episode=index,event=i,phase='training' if learn else 'fixed_dev',optimizer_updates=updates,elapsed_seconds=time.monotonic()-start))
  return np.asarray(preds),np.asarray(muscles)
 try:
  initial,_=rollout(2,stop=64);error=float(abs(initial-source_dev[:64]).max())
  if error>2e-6:raise ValueError('Zero current correction changed source prefix')
  arrays={};train=[]
  for i in [0,1]:
   pred,mu=rollout(i,True);arrays[f'online_train_{i}']=pred;train.append(command_metrics(pred,items[i][0]['labels'],items[i][0]['fit_mask']))
  pred,mu=rollout(2);arrays['dev']=pred;arrays['dev_muscles']=mu
  for name,value in model.state_dict().items():
   original=base['model'][name]
   if value.layout!=torch.strided:
    x,y=value.to_sparse_coo().coalesce(),original.to_sparse_coo().coalesce();same=torch.equal(x.indices(),y.indices()) and torch.equal(x.values(),y.values())
   else:same=torch.equal(value,original)
   if not same:raise ValueError('Changed old model '+name)
  for module,key in [(head,'readout'),(dn,'adapter'),(optical,'optical')]:
   for k,v in module.state_dict().items():
    if not torch.equal(v,cp[key][k]):raise ValueError('Changed frozen source '+key+'.'+k)
  cp.update(port_correction=correction.state_dict(),port_correction_parent_sha256=sr['checkpoint_sha256'],optimizer_updates=updates,actor_qualified=False)
  file=run/'steering.pt';torch.save(cp,file);np.savez_compressed(run/'predictions.npz',**arrays)
  out.update(status='completed_visual_port_correction_fit',optimizer_updates=updates,online_train=train,online_train_is_not_fixed_checkpoint_evaluation=True,dev=command_metrics(pred,items[2][0]['labels'],items[2][0]['fit_mask']),initial_source_prefix_max_error=error,checkpoint_sha256=sha256(file),predictions_sha256=sha256(run/'predictions.npz'),old_model_state_entries_exact=51,all_source_adapters_and_readout_exact=True,termination_reason='One fixed epoch, frozen head/source, full fixed development evaluation complete')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),optimizer_updates=updates,termination_reason='Explicit four-port correction failure')
 finally:h.remove();oh.remove()
 out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
