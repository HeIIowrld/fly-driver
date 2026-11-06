"""Continuous learning of separate visual turn and body-modulated speed intents."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.turn_speed_experiment import TurnSpeedExperiment,parameter_vector,gradient_vector
from scripts.continuous_descending_experiment import detach_exact,real_window,isolated_synthetic
from scripts.sac_steering_experiment import snapshot
from scripts.intervene_motion_spatial_same_body import state_digest

class ParentStateAccess:
 def __init__(self,values,allowed):self.values=values;self.allowed=allowed;self.reads=0;self.last_sha256=None
 def __getitem__(self,key):
  if not self.allowed:raise ValueError('Continuous arm attempted to inject parent numerical state')
  self.reads+=1;value=self.values[key];self.last_sha256=state_digest(value);return value

@torch.no_grad()
def project_speed_steps(parameters,old,plan):
 for index,key in [(1,'maximum_speed_step_norm'),(2,'maximum_body_gain_step_norm')]:
  delta=parameters[index]-old[index];factor=(plan[key]/delta.norm().clamp_min(1e-30)).clamp_max(1.);parameters[index].copy_(old[index]+delta*factor)

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--arm',choices=['turn_only','turn_speed'],required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(a.plan);assert sha256(plan['speed_gradient_validation_result'])==plan['speed_gradient_validation_result_sha256'] and read(plan['speed_gradient_validation_result'])['gradient_validated'];assert sha256(plan['neutral_validation_result'])==plan['neutral_validation_result_sha256'] and read(plan['neutral_validation_result'])['maximum_prediction_difference']<2e-6;assert sha256(plan['reserved_test_plan'])==plan['reserved_test_plan_sha256'];torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);e=TurnSpeedExperiment(plan,speed_active=a.arm=='turn_speed');synthetic=[dict(source_slot=s,pair=k,temperature=t,group=3*(k//2)+j) for s in range(3) for k in range(0,12,2) for j,t in enumerate(e.sources[0]['plan']['temperature_c'])];rng.shuffle(synthetic);pending=dict(zip(map(int,np.linspace(0,len(e.real_tasks)-1,len(synthetic),dtype=int)),synthetic));selected={(t['episode'],t['begin']):t for t in e.real_tasks};e.states=ParentStateAccess(e.states,False);out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),parent_candidate=e.parent,entries=e.entries,real_windows=e.real_tasks,train_episodes=[0,1],development_episode_used_for_fit=False,seed=plan['seed'],actor_qualified=False,vehicle_control_sent=False,scope=__doc__));write_json(out/'backend-initialization.json',e.backend)
 for name in plan['source_sha256']:
  target=out/'executed-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
 weight=e.dn.projection.weight;parameters=e.train_parameters();optimizer=torch.optim.Adam([{'params':[parameters[0]],'lr':plan['learning_rate']},{'params':[parameters[1]],'lr':plan['speed_learning_rate']},{'params':[parameters[2]],'lr':plan['body_gain_learning_rate']}]);records=[];before=[];after=[];gradients=[];proposals=[];journal=[];schedule=[];online={};selected_index=0;real_events=0;start=time.perf_counter();enabled=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
 def update(kind,task,parts,prediction):
  static,_,native,native_valid=e.inputs();speed_inputs=(static,e.last_speed['body_speed'],torch.ones(1,1,dtype=torch.bool),native,e.last_speed_native_valid,e.views,e.goals)
  with torch.no_grad():old_target=e.speed(*speed_inputs).clone()
  loss=sum(parts);optimizer.zero_grad(set_to_none=True);loss.backward();turn_norm=torch.nn.utils.clip_grad_norm_([parameters[0]],1.);speed_norm=torch.nn.utils.clip_grad_norm_(parameters[1:],1.);norm=torch.maximum(turn_norm,speed_norm);assert torch.isfinite(norm);old=[p.detach().clone() for p in parameters];before.append(parameter_vector(parameters));gradients.append(gradient_vector(parameters));cut=detach_exact(e.model);optimizer.step();proposal=parameter_vector(parameters);project_speed_steps(parameters,old,plan);e.project();after.append(parameter_vector(parameters));proposals.append(proposal)
  with torch.no_grad():speed_change=float(abs(e.speed(*speed_inputs)-old_target).max())
  assert speed_change<=plan['maximum_same_observation_speed_change_m_s']+1e-5
  names=['supervision','anchor','protection'] if kind=='real' else ['direction','anchor'];record=dict(kind=kind,task=task,loss=float(loss.detach()),parts={n:float(v.detach()) for n,v in zip(names,parts)},gradient_norm=float(norm),same_observation_speed_change_m_s=speed_change,turn_gradient_norm=float(turn_norm),speed_gradient_norm=float(speed_norm),detach_preserved_numeric_state_sha256=cut);records.append(record);schedule.append((kind,task));return record
 try:
  for episode in (0,1):
   e.model.reset_episode();d=e.items[episode][0];sequence=[]
   for begin in range(0,len(d['scene']),plan['window']):
    end=min(begin+plan['window'],len(d['scene']));task=selected.get((episode,begin));learn=task is not None;prior=e.model.previous_time;incoming=state_digest(snapshot(e.model));parts,prediction=real_window(e,episode,begin,end,learn=learn,parent_prefix=False);sequence.append(prediction.detach().numpy());real_events+=end-begin
    if learn:record=update('real',task,parts,prediction);cut=record['detach_preserved_numeric_state_sha256']
    else:cut=detach_exact(e.model)
    row=dict(episode=episode,begin=begin,end=end,selected=learn,parent_state_injected=False,previous_time_before=prior,previous_time_after=e.model.previous_time,incoming_numeric_sha256=incoming,start_numeric_sha256=incoming,detach_numeric_sha256=cut)
    if learn:
     if selected_index in pending:
      _,digest=isolated_synthetic(e,pending[selected_index],update);row['synthetic_resume_sha256']=digest
     selected_index+=1
    journal.append(row);write_json(out/'progress.json',dict(real_events=real_events,total_real_events=sum(len(e.items[k][0]['scene']) for k in (0,1)),optimizer_updates=len(records),elapsed_seconds=time.perf_counter()-start))
   online[f'prediction_{episode}']=np.concatenate(sequence)
  assert len(records)==147 and selected_index==93 and real_events==32597 and e.states.reads==0;e.verify_fixed();model,head,dn,optical,body_port,_=e.components;cp=out/'turn-speed.pt';torch.save(dict(schema='flybrain.turn_speed_intent.v1',speed_intent=e.speed.state_dict(),speed_active=e.speed_active,speed_limits=dict(max_delta_m_s=e.speed.max_delta_m_s,maximum_log_gain=e.speed.maximum_log_gain),parent_candidate=e.parent,model=model.state_dict(),motion_port=e.motion.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),optical=optical.state_dict(),body_port=body_port.state_dict(),optimizer_updates=147,actor_qualified=False,training_regime=a.arm,synthetic_gradient_mode='episode_full'),cp);np.savez_compressed(out/'optimizer-trajectory.npz',parameter_before=np.asarray(before),parameter_after=np.asarray(after),adam_proposal=np.asarray(proposals),applied_gradients=np.asarray(gradients));np.savez_compressed(out/'online-predictions.npz',**online);write_json(out/'training-records.json',records);write_json(out/'training-schedule.json',schedule);write_json(out/'state-journal.json',journal);result.update(status='completed_turn_speed_intent_fit',arm=a.arm,optimizer_updates=147,real_updates=93,synthetic_updates=54,real_events=real_events,real_episode_resets=2,synthetic_episode_resets=54,parent_state_injections=e.states.reads,checkpoint_sha256=sha256(cp),training_records_sha256=sha256(out/'training-records.json'),optimizer_trajectory_sha256=sha256(out/'optimizer-trajectory.npz'),state_journal_sha256=sha256(out/'state-journal.json'),online_predictions_sha256=sha256(out/'online-predictions.npz'),maximum_weight_change=float(abs(weight.detach()-e.initial_weight).max()),maximum_speed_weight=float(e.speed.visual_projection.weight.detach().abs().max()),maximum_body_gain_weight=float(e.speed.body_gain.weight.detach().abs().max()),trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),speed_active=e.speed_active,original_graph_body_readout_and_other_visual_weights_exact=True,parent_prefix_reused=False,parent_prefix_states_sha256=sha256(e.prefix_file),reserved_pixels_read=False,termination_reason='Chronological32597actual events/147updates; existing turn320 and optional visual speed320/native gain38. Own continuous state, separate bounded-speed trust steps. Gradient-only cuts exact, independent synthetic episodes resume owned real state. No authority/promotion.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),optimizer_updates=len(records),real_events=real_events)
 finally:e.close();gc.enable() if enabled else gc.disable()
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
