"""Recover exact Adam moments from recorded gradients without new model fitting."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/spatial-motion-dn-learning-20261104')

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');reports={}
 for arm in ['visual_only','visual_native']:
  fit=ROOT/f'{arm}-fit-01';r=read(fit/'result.json');assert r['status']=='completed_spatial_motion_dn_fit' and sha256(fit/'ports.pt')==r['checkpoint_sha256'] and sha256(fit/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);records=read(fit/'training-records.json');parameters=[torch.nn.Parameter(cp['current_calibration']['motion_projection.weight'].clone())]+([torch.nn.Parameter(cp['body_port']['projection.weight'].clone())] if arm=='visual_native' else []);rates=[plan['visual_learning_rate']]+([plan['native_learning_rate']] if arm=='visual_native' else []);optimizer=torch.optim.Adam([dict(params=[p],lr=lr) for p,lr in zip(parameters,rates)]);out=begin_run(fit/'optimizer-resume-01',dict(program_sha256=sha256(__file__),source_checkpoint_sha256=r['checkpoint_sha256'],source_trajectory_sha256=r['optimizer_trajectory_sha256'],seed=None,new_model_optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));maximum=0.;sizes=[p.numel() for p in parameters]
  with np.load(fit/'optimizer-trajectory.npz') as z:
   assert len(records)==len(z['before'])==395
   for j,row in enumerate(records):
    offset=0
    for k,(p,size) in enumerate(zip(parameters,sizes)):
     with torch.no_grad():p.copy_(torch.tensor(z['before'][j,offset:offset+size].reshape(p.shape)))
     p.grad=None if k==1 and row['native_grad_is_none'] else torch.tensor(z['gradients'][j,offset:offset+size].reshape(p.shape));offset+=size
    optimizer.step();proposal=torch.cat([p.detach().flatten() for p in parameters]).numpy();maximum=max(maximum,float(abs(proposal-z['proposal'][j]).max()))
  assert maximum==0;state=optimizer.state_dict();dest=out/'adam-state.pt';torch.save(state,dest);steps=[int(optimizer.state[p]['step'].item()) for p in parameters];assert steps==([395,331] if arm=='visual_native' else [395]);result=dict(status='completed_exact_adam_state_reconstruction',maximum_recorded_proposal_difference=maximum,steps=steps,source_checkpoint_sha256=r['checkpoint_sha256'],adam_state_sha256=sha256(dest),new_model_optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Recorded clipped gradients replayed with authentic pre-step weights and native grad=None skips. Every original Adam proposal matched bitwise; moments and steps saved for genuine continuation. Source checkpoints unchanged; this is optimizer-state reconstruction, no new training.');write_json(out/'result.json',result);reports[arm]=result
 write_json(ROOT/'optimizer-resume-receipt-01.json',dict(status='completed_exact_optimizer_resume_preparation',models=reports,new_model_optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));print({k:v['steps'] for k,v in reports.items()})
if __name__=='__main__':main()
