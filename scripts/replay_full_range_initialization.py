"""Whole-episode TRAIN warm registration; no fitted or development-selected initializer."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.full_range_port_experiment import FullRangePortExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.improve_sac_anticipation import command_metrics

ROOT=Path('runs/full-range-port-learning-20261104')
def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=[0,1],required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=FullRangePortExperiment(plan,train_body=True);episode=a.episode;tasks=[v for v in plan['physical_tasks'] if v['episode']==episode];begins={v['begin'] for v in tasks};d=e.items[episode][0];e.holder.update(mode='real',episode=episode);out=begin_run(ROOT/f'initial-episode{episode}-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),initial_gate=plan['initial_gate'],seed=None,optimizer_updates=0,development_used=False,actor_qualified=False,vehicle_control_sent=False));states={};sequence=[];muscles=[];saved={};reports=[];start=time.perf_counter()
 try:
  e.model.reset_episode()
  for i in range(len(d['scene'])):
   if i in begins:states[i]=snapshot(e.model)
   sequence.append(e.step(i,False).numpy());muscles.append(e.last_muscles[0].numpy())
   if i%1000==0:write_json(out/'progress.json',dict(events=i,total=len(d['scene']),elapsed_seconds=time.perf_counter()-start))
  pred=np.asarray(sequence);saved['prediction']=pred;saved['muscles']=np.asarray(muscles);torch.save(states,out/'prefix-states.pt')
  for j,t in enumerate(tasks):
   neutral=pred[t['begin']:t['end']];row=dict(task=t,commands={})
   for field in ['yaw','speed']:
    branches=[]
    for sign in [-1,1]:
     restore(e.model,states[t['begin']]);e.yaw_offset=sign*.01 if field=='yaw' else 0.;e.speed_offset=sign*.01 if field=='speed' else 0.;branches.append(np.asarray([e.step(i,False).numpy() for i in range(t['begin'],t['end'])]))
    e.yaw_offset=e.speed_offset=0.;p=np.asarray(branches);saved[f'case{j}_{field}_prediction']=p;v=p[:,:,0] if field=='yaw' else p[:,:,1]-p[:,:,2];base=neutral[:,0] if field=='yaw' else neutral[:,1]-neutral[:,2];s=-1 if field=='yaw' else 1;minus=v[0]-base;plus=v[1]-base;row['commands'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),both_correct=bool(s*minus.mean()<0 and s*plus.mean()>0))
   reports.append(row)
  with np.load('runs/motion-spatial-learning-20261104/fine_motion-fit-01/predictions.npz') as z:original=z[f'prediction_{episode}'].copy()
  e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_continuous_bilateral_registration_initialization',episode=episode,events=len(pred),reports=reports,metrics={role:dict(original=command_metrics(original,d['labels'],m),warm=command_metrics(pred,d['labels'],m)) for role,m in [('all',d['fit_mask']),('pre',e.pre[episode])]},yaw_both_correct=sum(v['commands']['yaw']['both_correct'] for v in reports),speed_both_correct=sum(v['commands']['speed']['both_correct'] for v in reports),predictions_sha256=sha256(out/'predictions.npz'),prefix_states_sha256=sha256(out/'prefix-states.pt'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,termination_reason='Predeclared .01 bilateral gate, complete continuous TRAIN history and identical-prefix command branches. No initializer selection from development or reserved pixels.'))
 finally:e.yaw_offset=e.speed_offset=0.;e.close()
if __name__=='__main__':main()
