"""Independent continued Adam, projections and numerical-state-chain audit."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 p=read(ROOT/'plan-01.json');source=torch.load(p['normalized_parent_optimizer'],map_location='cpu',weights_only=True);origin=torch.load(p['motion_parent_checkpoint'],map_location='cpu',weights_only=True)['body_port']['projection.weight'].numpy().reshape(-1).astype(float);reports={}
 for arm in ['neutral_control','speed_policy']:
  path=ROOT/f'{arm}-fit-01';r=read(path/'result.json');assert r['status']=='completed_symmetric_speed_fit' and sha256(path/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'];records=read(path/'training-records.json');journal=read(path/'state-journal.json');assert len(records)==395 and len(journal)==1020 and sum(v['end']-v['begin'] for v in journal)==32597 and not any(v['parent_state_injected'] for v in journal)
  for a,b in zip(journal,journal[1:]):
   if a['episode']==b['episode'] and a['epoch']==b['epoch']:assert a['end']==b['begin'] and a['after_sha256']==b['incoming_sha256']
  groups=[(0,8192,p['visual_learning_rate'],p['visual_step_norm']),(8192,17438,p['native_learning_rate'],p['native_step_norm'])]+([(17438,17445,p['speed_weight_learning_rate'],p['speed_weight_step_norm']),(17445,17446,p['speed_bias_learning_rate'],p['speed_bias_step_norm'])] if arm=='speed_policy' else []);moments=[]
  for k,(start,stop,lr,bound) in enumerate(groups):
   if k<2:
    v=source['state'][source['param_groups'][k]['params'][0]];moments.append((v['exp_avg'].numpy().reshape(-1).astype(float),v['exp_avg_sq'].numpy().reshape(-1).astype(float),int(v['step'])))
   else:moments.append((np.zeros(stop-start),np.zeros(stop-start),0))
  maximum=0.;maximum_ratio=0.;native_fixture=0.;policy_fixture=0.
  with np.load(path/'optimizer-trajectory.npz') as z:
   before=z['before'].astype(float);proposal=z['proposal'].astype(float);after=z['after'].astype(float);gradient=z['gradients'].astype(float)
   for j,row in enumerate(records):
    assert len(row['parameter_grad_is_none'])==len(groups)
    for k,(start,stop,lr,bound) in enumerate(groups):
     if row['parameter_grad_is_none'][k]:expected=before[j,start:stop]
     else:
      first,second,step=moments[k];step+=1;g=gradient[j,start:stop];first=.9*first+.1*g;second=.999*second+.001*g*g;moments[k]=(first,second,step);expected=before[j,start:stop]-lr*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
     maximum=max(maximum,float(abs(expected-proposal[j,start:stop]).max()));stepnorm=float(np.linalg.norm(after[j,start:stop]-before[j,start:stop]));maximum_ratio=max(maximum_ratio,stepnorm/bound);assert stepnorm<=bound+1e-7
    if row['kind'] in ['synthetic','image_exchange']:
     assert row['native_grad_is_none'] and row['speed_policy_grad_is_none'];native_fixture=max(native_fixture,float(abs(after[j,8192:17438]-before[j,8192:17438]).max()))
     if arm=='speed_policy':policy_fixture=max(policy_fixture,float(abs(after[j,17438:]-before[j,17438:]).max()))
   assert np.linalg.norm(after[:,:8192],axis=1).max()<=p['visual_total_norm']+1e-7 and np.linalg.norm(after[:,8192:17438]-origin,axis=1).max()<=p['native_total_norm']+1e-7 and abs(after[:,8192:17438]-origin).max()<=p['native_entry_delta']+1e-7
   if arm=='speed_policy':assert (after[:,17444]<=0).all() and np.linalg.norm(after[:,17438:17445],axis=1).max()<=p['speed_weight_total_norm']+1e-7 and abs(after[:,17445]).max()<=p['speed_bias_total_norm']+1e-7
  assert maximum<2e-7 and native_fixture==policy_fixture==0;steps=[v[2] for v in moments];assert steps==r['final_optimizer_steps'];reports[arm]=dict(updates=395,initial_steps=p['initial_optimizer_steps'],final_steps=steps,maximum_independent_adam_proposal_error=maximum,maximum_step_bound_ratio=maximum_ratio,maximum_native_fixture_change=native_fixture,maximum_speed_policy_fixture_change=policy_fixture,numerical_windows=1020,parent_state_injections=0)
 write_json(ROOT/'optimizer-audit-01.json',dict(status='completed_independent_flight_speed_policy_optimizer_audit',models=reports,actor_qualified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
