"""Fresh development command perturbations from a verified full parent trajectory."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.functional_priority_experiment import Experiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest

def main():
 torch.set_num_threads(1);root=Path('runs/paired-muscle-decoder-20261104');e=Experiment(read('runs/functional-priority-learning-20261104/fit-plan-01.json'));plan=read(root/'plan-01.json');assert e.parent==plan['parent_candidate'];d=e.items[2][0];blocks=np.unique(np.flatnonzero(e.pre[2])//32);selected=blocks[np.linspace(0,len(blocks)-1,3,dtype=int)];tasks=[dict(episode=2,begin=int(b*32),end=min(int((b+1)*32),len(d['scene']))) for b in selected];out=begin_run(root/'development-probes-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(root/'plan-01.json'),parent_candidate=e.parent,tasks=tasks,source_episode=e.entries[2],actor_qualified=False,vehicle_control_sent=False));e.model.reset_episode();e.holder.update(mode='real',episode=2);states={};pred=[]
 try:
  for i in range(len(d['scene'])):
   if any(t['begin']==i for t in tasks):states[i]=snapshot(e.model)
   pred.append(e.step(i,False).numpy())
   if i%1000==0:write_json(out/'progress.json',dict(events=i,total=len(d['scene'])))
  with np.load(Path(plan['parent_fine_fit'])/'predictions.npz') as z:baseline=z['prediction_2'].copy()
  parity=float(abs(np.asarray(pred)-baseline).max());assert parity<2e-6;offset=0.;saved={};mu=[]
  def change(module,args):
   command,body,valid,goals=args[:4];target=torch.minimum((command[:,1:]+offset).clamp_min(0),goals[:,1:]*100/3.6);return (torch.cat([command[:,:1],target],1),*args[1:])
  def record(module,args):mu.append(args[0][0].numpy().copy())
  hooks=[e.model.motor.command.register_forward_pre_hook(change),e.head.register_forward_pre_hook(record)]
  try:
   for k,task in enumerate(tasks):
    for step in (.01,.001):
     values=[];muscles=[]
     for sign in (-1,1):
      offset=sign*step;restore(e.model,states[task['begin']]);mu.clear();values.append(np.asarray([e.step(i,False).numpy() for i in range(task['begin'],task['end'])]));muscles.append(np.asarray(mu))
     saved[f'case_{k}_step_{step}_prediction']=np.asarray(values);saved[f'case_{k}_step_{step}_muscles']=np.asarray(muscles)
  finally:
   for hook in hooks:hook.remove()
  np.savez_compressed(out/'probes.npz',**saved);torch.save(states,out/'prefix-states.pt');write_json(out/'result.json',dict(status='completed_fresh_development_muscle_command_probes',full_episode_events=len(d['scene']),parent_full_prediction_parity=parity,tasks=tasks,initial_prefix_state_sha256={str(k):state_digest(v) for k,v in states.items()},raw_probes_sha256=sha256(out/'probes.npz'),prefix_states_sha256=sha256(out/'prefix-states.pt'),optimizer_updates=0,used_for_fit=False,actor_qualified=False,vehicle_control_sent=False,scope='Fresh derivative queries in already-consumed same-course development; full original recurrence verified first. Not new-track/actual control or physiological evidence.'))
 finally:e.close()
if __name__=='__main__':main()
