"""Independent Adam/projection, state continuity and fixture-gradient audit."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read

ROOT=Path('runs/full-range-port-learning-20261104')
def main():
 plan=read(ROOT/'plan-01.json');reports={}
 for arm in ['visual_only','visual_native']:
  fit=ROOT/f'{arm}-fit-01';r=read(fit/'result.json');assert r['status']=='completed_full_range_port_fit' and sha256(fit/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'] and sha256(fit/'training-records.json')==r['training_records_sha256'];records=read(fit/'training-records.json');journal=read(fit/'state-journal.json');expected_real=len(plan['selected_real_windows'])*plan['epochs'];assert r['real_updates']==expected_real and r['physical_updates']==30*plan['epochs'] and r['synthetic_updates']==54*plan['epochs'];assert sum(v['kind']=='image_exchange' for v in records)==10*plan['epochs'];assert len(journal)==1020*plan['epochs'] and sum(v['end']-v['begin'] for v in journal)==32597*plan['epochs'] and not any(v['parent_state_injected'] for v in journal)
  for previous,current in zip(journal,journal[1:]):
   if previous['epoch']==current['epoch'] and previous['episode']==current['episode']:assert previous['after_sha256']==current['incoming_sha256'] and previous['end']==current['begin']
  with np.load(fit/'optimizer-trajectory.npz') as z:
   before=z['before'].astype(float);proposal=z['proposal'].astype(float);after=z['after'].astype(float);gradient=z['gradients'].astype(float);n=before.shape[1];assert len(records)==len(before);groups=[(0,1280,plan['visual_learning_rate'],plan['visual_step_norm']),(1280,1284,plan['bias_learning_rate'],plan['bias_step_norm']),(1284,1288,plan['gate_learning_rate'],plan['gate_step_norm'])]+([(1288,n,plan['native_learning_rate'],plan['native_step_norm'])] if arm=='visual_native' else []);maximum_adam_error=0.;maximum_step=0.;native_fixture_changes=0.;moments=[(np.zeros(stop-start),np.zeros(stop-start),0) for start,stop,lr,bound in groups]
   for j,row in enumerate(records):
    for k,(start,stop,lr,bound) in enumerate(groups):
     if k==3 and row['native_grad_is_none']:
      expected=before[j,start:stop]
     else:
      first,second,step=moments[k];step+=1;grad=gradient[j,start:stop];first=.9*first+.1*grad;second=.999*second+.001*grad*grad;moments[k]=(first,second,step);expected=before[j,start:stop]-lr*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
     maximum_adam_error=max(maximum_adam_error,float(abs(expected-proposal[j,start:stop]).max()));delta=float(np.linalg.norm(after[j,start:stop]-before[j,start:stop]));maximum_step=max(maximum_step,delta/bound);assert delta<=bound+1e-7
    if row['kind'] in ['synthetic','image_exchange']:
     assert row['native_grad_is_none'] and row['native_parameter_change']==0
     if arm=='visual_native':native_fixture_changes=max(native_fixture_changes,float(abs(after[j,1288:]-before[j,1288:]).max()))
   assert maximum_adam_error<2e-7 and native_fixture_changes==0;reports[arm]=dict(updates=len(records),real_updates=r['real_updates'],physical_updates=r['physical_updates'],synthetic_updates=r['synthetic_updates'],image_updates=10*plan['epochs'],maximum_independent_adam_error=maximum_adam_error,maximum_step_bound_ratio=maximum_step,maximum_native_fixture_change=native_fixture_changes,numerical_chain_windows=len(journal),current_weight_change_norm=r['current_weight_change_norm'],native_weight_change_norm=r['native_weight_change_norm'])
 write_json(ROOT/'optimizer-audit-01.json',dict(status='completed_independent_full_range_optimizer_audit',models=reports,actor_qualified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
