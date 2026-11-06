"""Actual sensed-body feedback perturbations at fixed desired command and images."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment
from scripts.sac_steering_experiment import restore
from scripts.train_qualified_sac_unified import context

ROOT=Path('runs/fixed-full-routing-learning-20261104')
class FeedbackExperiment(FixedFullRoutingExperiment):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.feedback_yaw=0.;self.feedback_speed=0.;self.fixed_speed=None;self.command_error=0.;self.hooks.append(self.model.motor.command.register_forward_pre_hook(self._hold_command))
 def _hold_command(self,module,args):
  command=args[0].clone();command[:,0]=0.;command[:,1]=self.fixed_speed;self.command_error=max(self.command_error,float(abs(command[:,1]-self.fixed_speed).max()));return (command,*args[1:])
 def step(self,i,learn):
  if learn:raise ValueError('Body feedback test is read-only')
  self.holder['index']=i;d,c,_=self.items[self.holder['episode']];ctx=context(d,c,i);self.fixed_speed=float(d['body'][i,16])
  def changed(obs):
   if obs is None:return None
   values=obs.values.copy();assert obs.valid[[4,16]].all();values[4]+=self.feedback_yaw;values[16]+=self.feedback_speed;assert values[16]>=0;return replace(obs,values=values)
  ctx=per_view_temporal_body_context(tuple(changed(x) for x in ctx.visual_contexts),changed(ctx.decision_body),render_start_lowers=ctx.render_start_lowers,camera_receipt_time=ctx.camera_receipt_time,decision_time=ctx.decision_time)
  with torch.no_grad():
   _,mu=self.model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=self.goals,scene_features=torch.tensor(d['scene'][i:i+1]),scene_valid=self.views,body_context=ctx);return self.head(mu)[0]

def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=FeedbackExperiment(p,train_body=True);fit=ROOT/'visual_native-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);path=ROOT/'visual_native-evaluation-01';r=read(path/'result.json');assert sha256(path/'candidate-prefix-states.pt')==r['prefix_states_sha256'];states=torch.load(path/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);tasks=read('runs/multicorner-command-20261104/protocol-01.json')['tasks'];out=begin_run(ROOT/'body-feedback-probe-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'ports.pt'),prefix_sha256=r['prefix_states_sha256'],seed=None,optimizer_updates=0,scope='Perturb current and all historical sensed body yaw-left or speed by +/-.01 with same visual features/native19 and candidate prefix. Hold desired yaw=0 and desired speed at the original neutral source value exactly. Desired-command injection tests are separate. Expected body-yaw damping is opposite the desired-yaw test; speed feedback opposes sensed speed. Model-local feedback sign test, not physical yaw/vehicle dynamics, physiology or adaptation.',actor_qualified=False,vehicle_control_sent=False));reports=[];saved={};maximum_neutral=0.;counts={role:dict(total=0,yaw=0,speed=0) for role in ['TRAIN','development']}
 try:
  with np.load(path/'predictions.npz') as z:
   for j,t in enumerate(tasks):
    episode,begin,end=t['episode'],t['begin'],t['end'];e.holder.clear();e.holder.update(mode='real',episode=episode);values={}
    for name,yaw,speed in [('neutral',0.,0.),('yaw_minus',-.01,0.),('yaw_plus',.01,0.),('speed_minus',0.,-.01),('speed_plus',0.,.01)]:
     restore(e.model,states[episode,begin]);e.feedback_yaw=yaw;e.feedback_speed=speed;values[name]=np.asarray([e.step(i,False).numpy() for i in range(begin,end)]);saved[f'case{j}_{name}_prediction']=values[name]
    maximum_neutral=max(maximum_neutral,float(abs(values['neutral']-z[f'prediction_{episode}'][begin:end]).max()));role='TRAIN' if episode<2 else 'development';counts[role]['total']+=1;row=dict(task=t,feedback={})
    for field in ['yaw','speed']:
     axis=lambda v:v[:,0] if field=='yaw' else v[:,1]-v[:,2];neutral=axis(values['neutral']);minus=axis(values[field+'_minus'])-neutral;plus=axis(values[field+'_plus'])-neutral;expected=1 if field=='yaw' else -1;correct=bool(expected*minus.mean()<0 and expected*plus.mean()>0);counts[role][field]+=int(correct);row['feedback'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_both_damping=correct,mean_damping_and_each_change_above1e_minus5=bool(correct and abs(minus.mean())>1e-5 and abs(plus.mean())>1e-5),mean_minus_plus_slopes=[float(minus.mean()/.01),float(plus.mean()/.01)],correct_minus_fraction=float((expected*minus< -1e-7).mean()),correct_plus_fraction=float((expected*plus>1e-7).mean()))
    reports.append(row)
  assert maximum_neutral<2e-6 and e.command_error==0;e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_same_command_body_feedback_probe',reports=reports,counts=counts,maximum_neutral_replay_difference=maximum_neutral,maximum_desired_command_error=e.command_error,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,physical_feedback_stability_verified=False))
 finally:e.feedback_yaw=e.feedback_speed=0.;e.close()
if __name__=='__main__':main()
