"""Desired-left-yaw command probes at TRAIN and fresh development prefixes."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.functional_priority_experiment import Experiment
from scripts.sac_steering_experiment import restore
from scripts.intervene_motion_spatial_same_body import state_digest

def main():
 torch.set_num_threads(1);root=Path('runs/paired-muscle-decoder-20261104');plan=read(root/'plan-01.json');e=Experiment(read('runs/functional-priority-learning-20261104/fit-plan-01.json'));assert e.parent==plan['parent_candidate'];train=read(Path(plan['functional_probes'])/'result.json');development=read(root/'development-probes-01/result.json');devstates=torch.load(root/'development-probes-01/prefix-states.pt',map_location='cpu',weights_only=True);assert sha256(root/'development-probes-01/prefix-states.pt')==development['prefix_states_sha256'];tasks=[('TRAIN',j,v['task']) for j,v in enumerate(train['reports'])]+[('development',j,v) for j,v in enumerate(development['tasks'])];out=begin_run(root/'yaw-probes-01',dict(program_sha256=sha256(__file__),parent_candidate=e.parent,tasks=tasks,actor_qualified=False,vehicle_control_sent=False));offset=0.;mu=[];saved={};rows=[]
 def change(module,args):
  command=args[0];return (torch.cat([command[:,:1]+offset,command[:,1:]],1),*args[1:])
 def record(module,args):mu.append(args[0][0].numpy().copy())
 hooks=[e.model.motor.command.register_forward_pre_hook(change),e.head.register_forward_pre_hook(record)]
 with np.load(Path(plan['parent_fine_fit'])/'predictions.npz') as z:neutral={j:z[f'muscles_{j}'].copy() for j in range(3)}
 try:
  for role,j,task in tasks:
   episode=task['episode'];state=e.states[episode,task['begin']] if role=='TRAIN' else devstates[task['begin']];e.holder.update(mode='real',episode=episode);branches=[]
   for sign in (-1,1):
    restore(e.model,state);offset=sign*.01;mu.clear()
    for i in range(task['begin'],task['end']):e.step(i,False)
    branches.append(np.asarray(mu))
   saved[f'{role}_{j}_muscles']=np.asarray(branches);saved[f'{role}_{j}_neutral']=neutral[episode][task['begin']:task['end']];rows.append(dict(role=role,case=j,task=task,initial_state_sha256=state_digest(state)))
  np.savez_compressed(out/'probes.npz',**saved);write_json(out/'result.json',dict(status='completed_fixed_parent_yaw_command_probes',cases=rows,delta_yaw_left_rad_s=.01,raw_probes_sha256=sha256(out/'probes.npz'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Engineering positive-left-yaw field, held source body/images; canonical vehicle left steering is negative under existing recorded sign convention. Modeled muscle proxies, not measured wing force or actual vehicle yaw.'))
 finally:
  for h in hooks:h.remove()
  e.close()
if __name__=='__main__':main()
