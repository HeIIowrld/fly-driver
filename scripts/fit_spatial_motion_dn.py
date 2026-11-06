"""Chronological visual/native-port learning, frozen signed decoder and exact real state."""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.functional_priority_objective import recording_losses
from scripts.spatial_motion_dn_experiment import SpatialMotionDNExperiment as VisualNativePortExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.continuous_descending_experiment import detach_exact,isolated_synthetic
from scripts.intervene_motion_spatial_same_body import state_digest

ROOT=Path('runs/spatial-motion-dn-learning-20261104')

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['visual_only','visual_native'],required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');assert all(sha256(f)==h for f,h in plan['source_sha256'].items());validation=read(ROOT/'validation-01/result.json');assert validation['gradient_validated'] and validation['synthetic_native_grad_is_none'] and validation['synthetic_native_optimizer_change']==0;torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);e=VisualNativePortExperiment(plan,train_body=a.arm=='visual_native');parameters=e.train_parameters();optimizer=torch.optim.Adam([{'params':[parameter],'lr':rate} for parameter,rate in zip(parameters,e.learning_rates())]);selected={(t['episode'],t['begin']):t for t in plan['selected_real_windows']};synthetic=[dict(source_slot=s,pair=k,temperature=t,group=3*(k//2)+j) for s in range(3) for k in range(0,12,2) for j,t in enumerate(e.sources[0]['plan']['temperature_c'])];rng.shuffle(synthetic);pending=dict(zip(map(int,np.linspace(0,len(selected)-1,len(synthetic),dtype=int)),synthetic));out=begin_run(ROOT/f'{a.arm}-fit-01',dict(plan=plan,plan_sha256=sha256(ROOT/'plan-01.json'),arm=a.arm,parameters=sum(v.numel() for v in parameters),parent_candidate=e.parent,seed=plan['seed'],development_used_for_fit=False,reserved_pixels_read=False,scope=__doc__,actor_qualified=False,vehicle_control_sent=False));records=[];journal=[];before=[];proposals=[];after=[];gradients=[];online={};events=0;epoch_index=0;start=time.perf_counter();enabled=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
 def vector(values):return torch.cat([v.detach().reshape(-1) for v in values]).numpy().copy()
 def update(kind,task,parts,prediction=None):
  loss=sum(parts);optimizer.zero_grad(set_to_none=True);loss.backward();norms=[float(torch.nn.utils.clip_grad_norm_([v],1.)) for v in parameters];assert np.isfinite(norms).all();native_none=not e.train_body or parameters[e.native_parameter_index].grad is None
  if kind in ('synthetic','image_exchange') and not native_none:raise ValueError('Nonphysical image/body fixture created native body gradient')
  old=[v.detach().clone() for v in parameters];before.append(vector(parameters));gradients.append(np.concatenate([(v.grad.detach().numpy() if v.grad is not None else np.zeros(v.shape,dtype=np.float32)).reshape(-1) for v in parameters]));cut=detach_exact(e.model);optimizer.step();proposals.append(vector(parameters));e.project_after_step(old);after.append(vector(parameters));native_change=float((e.body_port.projection.weight-old[e.native_parameter_index]).abs().max()) if e.train_body else 0.
  if kind in ('synthetic','image_exchange') and native_change!=0:raise ValueError('Adam/projection moved native weights without real data gradient')
  row=dict(epoch=epoch_index,kind=kind,task=task,loss=float(loss.detach()),parts=[float(v.detach()) for v in parts],gradient_norms=norms,native_grad_is_none=native_none,native_parameter_change=native_change,parameter_step_norms=[float((v.detach()-old[k]).norm()) for k,v in enumerate(parameters)],detach_numeric_sha256=cut);records.append(row);return row
 def physical(task,prefix,formed_at):
  saved=snapshot(e.model);holder=e.holder.copy();expected=state_digest(saved);branches={}
  for name,yaw,speed in [('neutral',0.,0.),('yaw_minus',-.01,0.),('yaw_plus',.01,0.),('speed_minus',0.,-.01),('speed_plus',0.,.01)]:
   restore(e.model,prefix);e.holder.update(mode='real',episode=task['episode']);e.yaw_offset=yaw;e.speed_offset=speed;pred=[];logit=[]
   for i in range(task['begin'],task['end']):
    pred.append(e.step(i,True));logit.append(e.head.steering(e.head.features(e.last_muscles))[0,0])
   branches[name]=(torch.stack(pred),torch.stack(logit))
  e.yaw_offset=e.speed_offset=0.;neutral,logit0=branches['neutral'];minus=(branches['yaw_minus'][1]-logit0)/.01;plus=(branches['yaw_plus'][1]-logit0)/.01;q0=neutral[:,1]-neutral[:,2];speedminus=((branches['speed_minus'][0][:,1]-branches['speed_minus'][0][:,2])-q0)/.01;speedplus=((branches['speed_plus'][0][:,1]-branches['speed_plus'][0][:,2])-q0)/.01;loss=plan['physical_weight']*(torch.relu(plan['minimum_yaw_logit_slope']-minus.mean()).square()+torch.relu(plan['minimum_yaw_logit_slope']+plus.mean()).square()+torch.relu(-minus).square().mean()+torch.relu(plus).square().mean()+torch.relu(plan['minimum_speed_response_slope']+speedminus.mean()).square()+torch.relu(plan['minimum_speed_response_slope']-speedplus.mean()).square());restore(e.model,saved);e.holder.clear();e.holder.update(holder);assert state_digest(snapshot(e.model))==expected;detail=dict(task,prefix_formed_after_updates=formed_at,query_after_updates=len(records),prefix_sha256=state_digest(prefix),resumed_real_state_sha256=expected,yaw_logit_minus_plus=[float(minus.detach().mean()),float(plus.detach().mean())],speed_minus_plus=[float(speedminus.detach().mean()),float(speedplus.detach().mean())]);update('physical',detail,[loss])
 def image_exchange(task,prefix):
  saved=snapshot(e.model);holder=e.holder.copy();expected=state_digest(saved);e.holder.update(mode='real',episode=task['episode'],counterfactual_images=True);restore(e.model,prefix);e.holder.pop('donor',None)
  neutral=torch.stack([e.step(i,False) for i in range(task['begin'],task['end'])]);branches=[]
  for donor in [task['left_donor'],task['right_donor']]:
   restore(e.model,prefix);sequence=[]
   for offset,i in enumerate(range(task['begin'],task['end'])):e.holder['donor']=(task['episode'],donor+offset);sequence.append(e.step(i,True))
   branches.append(torch.stack(sequence))
  left,right=branches;margin=-(left[:,0]-right[:,0])/2;direction=plan['image_direction_weight']*torch.relu(plan['image_target_margin']-margin).square().mean();centre=plan['image_anchor_weight']*((.5*(left+right)-neutral).square()*left.new_tensor([4.,1.,1.])).sum(1).mean();restore(e.model,saved);e.holder.clear();e.holder.update(holder);assert state_digest(snapshot(e.model))==expected;update('image_exchange',dict(task,resumed_real_state_sha256=expected),[direction,centre])
 try:
  for epoch_index in range(plan['epochs']):
   selected_index=0
   for episode in [0,1]:
    e.model.reset_episode();d=e.items[episode][0];sequence=[];tasks=[t for t in plan['physical_tasks'] if t['episode']==episode];prefixes={};processed=set();e.holder.update(mode='real',episode=episode)
    for begin in range(0,len(d['scene']),plan['window']):
     end=min(begin+plan['window'],len(d['scene']));task=selected.get((episode,begin));learn=task is not None;incoming=state_digest(snapshot(e.model));prediction=[]
     for i in range(begin,end):
      for t in tasks:
       if i==t['begin']:prefixes[t['begin']]=(snapshot(e.model),len(records))
      prediction.append(e.step(i,learn))
     prediction=torch.stack(prediction);sequence.append(prediction.detach().numpy());events+=end-begin
     if learn:
      parts=recording_losses(prediction,torch.tensor(e.anchors[episode][begin:end]),torch.tensor(d['labels'][begin:end]),torch.tensor(d['fit_mask'][begin:end]),torch.tensor(e.pre[episode][begin:end]),anchor_weight=e.plan['real_anchor_weight'],maximum_ratio=e.plan['actual_recording_preservation_maximum_ratio'],protection_weight=e.plan.get('protection_weight',1.));update('real',task,parts,prediction)
     else:detach_exact(e.model)
     for t in tasks:
      if t['end']<=end and t['begin'] not in processed:
       physical(t,*prefixes[t['begin']]);processed.add(t['begin'])
       if t['phase']=='early':image_exchange(next(v for v in plan['image_tasks'] if v['episode']==episode and v['begin']==t['begin']),prefixes[t['begin']][0])
     if learn:
      if selected_index in pending:isolated_synthetic(e,pending[selected_index],update)
      selected_index+=1
     journal.append(dict(epoch=epoch_index,episode=episode,begin=begin,end=end,learn=learn,incoming_sha256=incoming,after_sha256=state_digest(snapshot(e.model)),parent_state_injected=False));write_json(out/'progress.json',dict(epoch=epoch_index,real_events=events,total_real_events=32597*plan['epochs'],updates=len(records),elapsed_seconds=time.perf_counter()-start))
    assert len(processed)==15;online[f'epoch{epoch_index}_prediction_{episode}']=np.concatenate(sequence)
   assert selected_index==len(selected)
  e.verify_fixed();cp=out/'ports.pt';torch.save(dict(schema='flybrain.spatial_motion_dn_learning.experimental.v1',arm=a.arm,current_calibration=e.calibration.state_dict(),body_port=e.body_port.state_dict(),fixed_decoder=plan['fixed_decoder'],fixed_decoder_sha256=plan['fixed_decoder_sha256'],parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],parent_port_checkpoint_sha256=plan['parent_port_checkpoint_sha256'],motion_parent_checkpoint_sha256=plan['motion_parent_checkpoint_sha256'],initial_gate=plan['initial_gate'],minimum_gate=plan['minimum_gate'],actor_qualified=False),cp);np.savez_compressed(out/'optimizer-trajectory.npz',before=np.asarray(before),proposal=np.asarray(proposals),after=np.asarray(after),gradients=np.asarray(gradients));np.savez_compressed(out/'online-predictions.npz',**online);write_json(out/'training-records.json',records);write_json(out/'state-journal.json',journal);result.update(status='completed_spatial_motion_dn_fit',optimizer_updates=len(records),real_events=events,real_updates=sum(v['kind']=='real' for v in records),physical_updates=sum(v['kind']=='physical' for v in records),synthetic_updates=sum(v['kind']=='synthetic' for v in records),checkpoint_sha256=sha256(cp),optimizer_trajectory_sha256=sha256(out/'optimizer-trajectory.npz'),training_records_sha256=sha256(out/'training-records.json'),state_journal_sha256=sha256(out/'state-journal.json'),online_predictions_sha256=sha256(out/'online-predictions.npz'),current_weight_change_norm=float((e.calibration.projection.weight-e.current_origin).norm()),native_weight_change_norm=float((e.body_port.projection.weight-e.native_origin).norm()),fixed_decoder_and_original_chemical_graph_exact=True,parent_state_injections=0,termination_reason='Declared chronological selected TRAIN supervision, independent same-prefix five-branch physical updates and full-history synthetic visual updates completed. Real state resumed exactly, no native gradient/momentum from synthetic fixtures, final weights retained without development/test selection.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),real_events=events,optimizer_updates=len(records))
 finally:e.yaw_offset=e.speed_offset=0.;e.close();gc.enable() if enabled else gc.disable()
 result['gate_values']=e.calibration.gate.detach().tolist();result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(result)
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
