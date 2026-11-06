"""Independently check every accepted TRAIN coefficient vector against original errors."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from scripts.fit_current_flight_launch_pedals import datasets

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);fit=Path(a.fit);m=read(fit/'manifest.json')['config'];r=read(fit/'result.json');out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),original_fit_result_sha256=sha256(fit/'result.json'),data_auditor_sha256=sha256('scripts/fit_current_flight_launch_pedals.py'),vehicle_control_sent=False));result=dict(status='failed',vehicle_control_sent=False,optimizer_updates=0)
 try:
  assert r['training_high_mse_limit_ratio']==m['training_high_mse_limit_ratio']==1.02;assert sha256(m['split'])==m['split_sha256'];assert sha256(fit/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'];items=datasets(Path(m['evaluation']));head,_=load_muscle_interaction_head(m['fixed_decoder']);ref=head.base.training_reference.numpy();scale=head.base.training_scale.numpy();groups=[]
  for v in items[:2]:
   idx=v['high'];z=((v['muscles'][idx]-ref)/scale).astype(np.float32);target=v['data']['labels'][idx,1:].astype(np.float32);baseline=((v['original'][idx,1:]-target)**2).mean(0);groups.append((z,target,baseline))
  with np.load(fit/'optimizer-trajectory.npz') as data:after=data['after'].copy();unique=np.unique(after,axis=0)
  worst=0.
  for weights in unique:
   w=weights[:18].reshape(3,6);b=weights[18:]
   for z,target,baseline in groups:
    logits=(z@w.T+b)[:,1:];pred=1/(1+np.exp(-np.clip(logits,-80,80)));mse=((pred-target)**2).mean(0);worst=max(worst,float((mse/baseline).max()));assert np.all(mse<=baseline*1.02+3e-8)
  result.update(status='completed_every_step_train_constraint_audit',steps=len(after),distinct_coefficient_vectors=len(unique),maximum_training_mse_ratio=worst,each_episode_and_pedal=True,development_used_for_fit=False,trajectory_sha256=sha256(fit/'optimizer-trajectory.npz'))
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 write_json(out/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
