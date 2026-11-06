"""Fixed factual command trace versus natural actor-policy body feedback."""
import argparse
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.direct_scene_gain_experiment import DirectSceneGainExperiment as FixtureAdjointSpeedExperiment
from scripts.sac_steering_experiment import restore
ROOT=Path('runs/direct-scene-gain-learning-20261105')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['frozen_direct','learn_direct'],required=True);a=parser.parse_args()
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=FixtureAdjointSpeedExperiment(p,learn_direct=a.arm=='learn_direct');fit=ROOT/f'{a.arm}-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);e.direct_gain.load_state_dict(cp['direct_gain']);e.direct_gain.requires_grad_(False);e.speed_policy.requires_grad_(False);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);path=ROOT/f'{a.arm}-evaluation-01';r=read(path/'result.json');assert sha256(path/'candidate-prefix-states.pt')==r['prefix_states_sha256'];states=torch.load(path/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);tasks=read('runs/multicorner-command-20261104/protocol-01.json')['tasks'];out=begin_run(ROOT/f'{a.arm}-body-feedback-probe-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'ports.pt'),prefix_sha256=r['prefix_states_sha256'],seed=None,optimizer_updates=0,scope='Current and all historical sensed yaw/speed perturbed +/-.01; image/native/clock and candidate prefix unchanged. Fixed mode holds this candidate own factual desired command trace, not measured speed. Policy mode lets the learned speed target update. Local model feedback signs, not physical driving stability.',actor_qualified=False,vehicle_control_sent=False));saved={};reports=[];counts={mode:dict(total=0,yaw=0,speed=0,nontrivial_yaw=0,nontrivial_speed=0) for mode in ['fixed_reference','actor_policy']};maximum_neutral=0.;maximum_reference=0.
 try:
  with np.load(path/'predictions.npz') as z:
   for j,t in enumerate(tasks):
    episode,begin,end=t['episode'],t['begin'],t['end'];trace=z[f'command_{episode}'][begin:end]
    for mode in counts:
     values={}
     for name,yaw,speed in [('neutral',0.,0.),('yaw_minus',-.01,0.),('yaw_plus',.01,0.),('speed_minus',0.,-.01),('speed_plus',0.,.01)]:
      restore(e.model,states[episode,begin]);e.holder.clear();e.holder.update(mode='real',episode=episode,sensed_yaw_delta=yaw,sensed_speed_delta=speed,case_begin=begin)
      if mode=='fixed_reference':e.holder['fixed_command_trace']=trace
      pred=[]
      for i in range(begin,end):
       pred.append(e.step(i,False).numpy())
       if mode=='fixed_reference':maximum_reference=max(maximum_reference,float(abs(e.last_command[0].numpy()-trace[i-begin]).max()))
      values[name]=np.asarray(pred);saved[f'case{j}_{mode}_{name}_prediction']=values[name]
     maximum_neutral=max(maximum_neutral,float(abs(values['neutral']-z[f'prediction_{episode}'][begin:end]).max()));counts[mode]['total']+=1;row=dict(task=t,mode=mode,feedback={})
     for field in ['yaw','speed']:
      axis=lambda x:x[:,0] if field=='yaw' else x[:,1]-x[:,2];minus=axis(values[field+'_minus'])-axis(values['neutral']);plus=axis(values[field+'_plus'])-axis(values['neutral']);sign=1 if field=='yaw' else -1;good=bool(sign*minus.mean()<0 and sign*plus.mean()>0);large=bool(good and abs(minus.mean())>1e-5 and abs(plus.mean())>1e-5);counts[mode][field]+=int(good);counts[mode]['nontrivial_'+field]+=int(large);row['feedback'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_both_damping=good,mean_damping_above1e_minus5=large)
     reports.append(row)
  assert maximum_neutral<2e-6 and maximum_reference==0;e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_fixture_adjoint_speed_body_feedback',reports=reports,counts=counts,maximum_neutral_replay_difference=maximum_neutral,maximum_factual_reference_error=maximum_reference,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,physical_feedback_stability_verified=False));print(counts)
 finally:e.close()
if __name__=='__main__':main()
