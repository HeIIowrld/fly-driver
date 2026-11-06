"""Continuous owned TRAIN recurrence with joint visual-DN and decoder updates."""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from scripts.continuous_descending_experiment import real_window,detach_exact,isolated_synthetic
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[1,4],required=True);a=p.parse_args();torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');assert all(sha256(p)==h for p,h in plan['source_sha256'].items());warm=root/'warm-fit-01';assert sha256(warm/'warm-head.pt')==read(warm/'result.json')['checkpoint_sha256'];torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);e=JointDNDecoderExperiment(plan,a.rank,torch.load(warm/'warm-head.pt',map_location='cpu',weights_only=True)['readout']);selected={(t['episode'],t['begin']):t for t in e.real_tasks};synthetic=[dict(source_slot=s,pair=k,temperature=t,group=3*(k//2)+j) for s in range(3) for k in range(0,12,2) for j,t in enumerate(e.sources[0]['plan']['temperature_c'])];rng.shuffle(synthetic);pending=dict(zip(map(int,np.linspace(0,len(e.real_tasks)-1,len(synthetic),dtype=int)),synthetic));data_plan=read(plan['paired_data_plan']);yawpath=Path(plan['yaw_probes'])/'probes.npz';speedpath=Path(data_plan['functional_probes'])/'probes.npz'
 with np.load(yawpath) as z:cached_yaw=[torch.tensor(z[f'TRAIN_{j}_muscles']) for j in plan['physical_cases']]
 with np.load(speedpath) as z:cached_speed=[torch.tensor(z[f'case_{j}_step_0.01_muscles']) for j in plan['physical_cases']]
 physical_tasks={(v['task']['episode'],v['task']['begin']) for v in read(Path(plan['yaw_probes'])/'result.json')['cases'] if v['role']=='TRAIN' and v['case'] in plan['physical_cases']};assert physical_tasks.issubset(selected)
 out=begin_run(root/f'rank{a.rank}-fit-01',dict(plan=plan,plan_sha256=sha256(root/'fit-plan-01.json'),rank=a.rank,parameters=sum(p.numel() for p in e.train_parameters()),parent_candidate=e.parent,warm_checkpoint_sha256=sha256(warm/'warm-head.pt'),entries=e.entries,real_windows=e.real_tasks,physical_tasks=sorted(physical_tasks),development_used_for_fit=False,reserved_pixels_read=False,actor_qualified=False,vehicle_control_sent=False));parameters=e.train_parameters();opt=torch.optim.Adam([{'params':[parameters[0]],'lr':plan['current_learning_rate']},{'params':parameters[1:],'lr':plan['head_learning_rate']}]);records=[];journal=[];events=0;selected_index=0;start=time.perf_counter();online={};result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False);enabled=gc.isenabled();gc.disable()
 def physical_loss(yaw,speed):
  yd=torch.stack([(v[1,:,0]-v[0,:,0]).mean()/.02 for v in yaw]);sd=torch.stack([((v[1,:,1]-v[1,:,2])-(v[0,:,1]-v[0,:,2])).mean()/.02 for v in speed]);return plan['physical_weight']*(torch.relu(yd+plan['minimum_physical_derivative']).square().mean()+torch.relu(plan['minimum_physical_derivative']-sd).square().mean()),yd,sd
 def cached():return physical_loss([e.head(mu.reshape(-1,6)).reshape(2,-1,3) for mu in cached_yaw],[e.head(mu.reshape(-1,6)).reshape(2,-1,3) for mu in cached_speed])
 def update(kind,task,parts,prediction,extra=None):
  penalty,yd,sd=cached();loss=sum(parts)+penalty
  if extra is not None:loss=loss+extra[0]
  opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm);cut=detach_exact(e.model);old=[v.detach().clone() for v in parameters];opt.step();e.project();records.append(dict(kind=kind,task=task,loss=float(loss.detach()),parts=[float(v.detach()) for v in parts],cached_physical_loss=float(penalty.detach()),cached_yaw=yd.detach().tolist(),cached_speed=sd.detach().tolist(),dynamic_physical=None if extra is None else dict(loss=float(extra[0].detach()),yaw=extra[1].detach().tolist(),speed=extra[2].detach().tolist()),gradient_norm=float(norm),parameter_change_norms=[float((v.detach()-b).norm()) for v,b in zip(parameters,old)],detach_numeric_sha256=cut));return records[-1]
 try:
  for episode in [0,1]:
   e.model.reset_episode();sequence=[];d=e.items[episode][0]
   for begin in range(0,len(d['scene']),plan['window']):
    end=min(begin+plan['window'],len(d['scene']));task=selected.get((episode,begin));learn=task is not None;incoming=snapshot(e.model);incoming_hash=state_digest(incoming);parts,prediction=real_window(e,episode,begin,end,learn=learn,parent_prefix=False);sequence.append(prediction.detach().numpy());events+=end-begin;extra=None
    if learn and (episode,begin) in physical_tasks:
     ending=snapshot(e.model);ending_hash=state_digest(ending);branches={}
     for field in ['yaw','speed']:
      pair=[]
      for sign in [-1,1]:
       restore(e.model,incoming);e.yaw_offset=sign*.01 if field=='yaw' else 0.;e.speed_offset=sign*.01 if field=='speed' else 0.;pair.append(torch.stack([e.step(i,True) for i in range(begin,end)]))
      branches[field]=[torch.stack(pair)]
     e.yaw_offset=e.speed_offset=0.;restore(e.model,ending);assert state_digest(snapshot(e.model))==ending_hash;extra=physical_loss(branches['yaw'],branches['speed'])
    if learn:update('real',task,parts,prediction,extra)
    else:detach_exact(e.model)
    if learn:
     if selected_index in pending:isolated_synthetic(e,pending[selected_index],update)
     selected_index+=1
    journal.append(dict(episode=episode,begin=begin,end=end,learn=learn,incoming_sha256=incoming_hash,after_sha256=state_digest(snapshot(e.model)),parent_state_injected=False));write_json(out/'progress.json',dict(real_events=events,total_real_events=32597,updates=len(records),elapsed_seconds=time.perf_counter()-start))
   online[f'prediction_{episode}']=np.concatenate(sequence)
  assert selected_index==93 and len(records)==147 and events==32597;e.verify_fixed();cp=out/'joint-dn-decoder.pt';torch.save(dict(schema='flybrain.joint_dn_decoder.experimental.v1',readout=e.head.state_dict(),current_calibration=e.calibration.state_dict(),rank=a.rank,parent_candidate=e.parent,actor_qualified=False),cp);np.savez_compressed(out/'online-predictions.npz',**online);write_json(out/'training-records.json',records);write_json(out/'state-journal.json',journal);result.update(status='completed_joint_dn_decoder_fit',real_events=events,optimizer_updates=len(records),real_updates=93,synthetic_updates=54,dynamic_physical_updates=sum(v['dynamic_physical'] is not None for v in records),parent_state_injections=0,checkpoint_sha256=sha256(cp),online_predictions_sha256=sha256(out/'online-predictions.npz'),training_records_sha256=sha256(out/'training-records.json'),state_journal_sha256=sha256(out/'state-journal.json'),maximum_current_weight=float(e.calibration.projection.weight.detach().abs().max()),fixed_original_components_exact=True,termination_reason='Chronological fixed TRAIN pass completed; final state retained without development/test selection. Numerical recurrence preserved across detach and isolated diagnostics/synthetic episodes.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),real_events=events,optimizer_updates=len(records))
 finally:e.yaw_offset=e.speed_offset=0.;e.close();gc.enable() if enabled else gc.disable()
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(result)
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
