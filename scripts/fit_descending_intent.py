"""Matched real-only versus joint learning of existing DN visual projection."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.descending_intent_experiment import DescendingExperiment

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--arm',choices=['real_only','joint'],required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(a.plan);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);diagnostic=read(plan['diagnostic_result']);assert sha256(plan['diagnostic_result'])==plan['diagnostic_result_sha256'] and diagnostic['gradient_validated'];assert sha256(plan['reserved_test_plan'])==plan['reserved_test_plan_sha256'];e=DescendingExperiment(plan);actual=list(e.real_tasks);synthetic=[dict(source_slot=s,pair=k,temperature=t,group=3*(k//2)+j) for s in range(3) for k in range(0,12,2) for j,t in enumerate(e.sources[0]['plan']['temperature_c'])];rng.shuffle(actual);rng.shuffle(synthetic);pending=dict(zip(map(int,np.linspace(0,len(actual)-1,len(synthetic),dtype=int)),synthetic));schedule=[]
 for j,task in enumerate(actual):
  schedule.append(('real',task))
  if j in pending:schedule.append(('synthetic',pending[j]))
 assert len(schedule)==147
 out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),parent_candidate=e.parent,entries=e.entries,real_windows=e.real_tasks,train_episodes=[0,1],development_episode_used_for_fit=False,seed=plan['seed'],actor_qualified=False,vehicle_control_sent=False,synthetic_gradient_mode='episode_full',scope=__doc__));write_json(out/'training-schedule.json',schedule);write_json(out/'backend-initialization.json',e.backend)
 for name in plan['source_sha256']:
  target=out/'executed-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
 weight=e.dn.projection.weight;optimizer=torch.optim.Adam([weight],lr=plan['learning_rate']);records=[];before=[];after=[];gradients=[];start=time.perf_counter();enabled=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
 try:
  for kind,task in schedule:
   if kind=='real':
    parts,_=e.real(task);loss=parts[0]+parts[1];names=['supervision','anchor','protection']
   else:
    parts,_=e.synthetic(task,'episode_full');loss=parts[1]+(parts[0] if a.arm=='joint' else 0);names=['direction','anchor']
   optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_([weight],1.);assert torch.isfinite(norm);before.append(weight.detach().numpy().copy());gradients.append(weight.grad.detach().numpy().copy());optimizer.step();e.project();after.append(weight.detach().numpy().copy());records.append(dict(kind=kind,task=task,loss=float(loss.detach()),parts={n:float(v.detach()) for n,v in zip(names,parts)},gradient_norm=float(norm)));write_json(out/'progress.json',dict(optimizer_updates=len(records),total=147,elapsed_seconds=time.perf_counter()-start))
  e.verify_fixed();model,head,dn,optical,body_port,_=e.components;cp=out/'descending-intent.pt';torch.save(dict(schema='flybrain.descending_intent.v1',parent_candidate=e.parent,model=model.state_dict(),motion_port=e.motion.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),optical=optical.state_dict(),body_port=body_port.state_dict(),optimizer_updates=len(records),actor_qualified=False,training_regime=a.arm,synthetic_gradient_mode='episode_full'),cp);np.savez_compressed(out/'optimizer-trajectory.npz',weight_before=np.asarray(before),weight_after=np.asarray(after),applied_gradients=np.asarray(gradients));write_json(out/'training-records.json',records);result.update(status='completed_descending_intent_fit',arm=a.arm,optimizer_updates=len(records),real_updates=93,synthetic_updates=54,checkpoint_sha256=sha256(cp),training_records_sha256=sha256(out/'training-records.json'),optimizer_trajectory_sha256=sha256(out/'optimizer-trajectory.npz'),maximum_weight_change=float(abs(weight-e.initial_weight).max()),original_graph_body_readout_and_other_visual_weights_exact=True,parent_prefix_reused=True,parent_prefix_states_sha256=sha256(e.prefix_file),reserved_pixels_read=False,termination_reason='147 matched updates; only existing320->1 engineering DN projection. Full synthetic history gradients, explicit original real numerical prefix cuts. Independent continuous evaluation required; no authority/promotion.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),optimizer_updates=len(records))
 finally:e.close();gc.enable() if enabled else gc.disable()
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
