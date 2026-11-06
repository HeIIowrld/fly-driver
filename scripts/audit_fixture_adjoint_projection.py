"""Reconstruct the new power weighted-L1 projection independently in NumPy."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/fixture-adjoint-learning-20261104')
def project(w,p):
 w=w.astype(np.float32).copy();s=np.array(p['speed_normalization']['scale'],dtype=np.float32);a=p['speed_max_delta_m_s']*np.array([1/float(s[2]),1/(2*float(s[5]))]);v=w[[2,5]].astype(float);maximum=p['maximum_power_input_sensitivity']
 if (a*abs(v)).sum()>maximum:
  low=0.;high=max(abs(v)/a)
  for _ in range(64):
   mid=(low+high)/2
   if (a*np.maximum(abs(v)-mid*a,0)).sum()>maximum:low=mid
   else:high=mid
  w[[2,5]]=(np.sign(v)*np.maximum(abs(v)-high*a,0)).astype(np.float32)
  sensitivity=np.float32(p['speed_max_delta_m_s'])*np.float32(abs(w[2])/s[2]+abs(w[5])/(np.float32(2)*s[5]))
  if float(sensitivity)>maximum:w[[2,5]]*=np.float32(maximum/float(sensitivity))
 return w

def main():
 p=read(ROOT/'plan-01.json');reports={}
 for arm in ['stop_all','input_adjoint']:
  path=ROOT/f'{arm}-fit-01';r=read(path/'result.json');assert sha256(path/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'];records=read(path/'training-records.json');maximum=0.;count=0
  with np.load(path/'optimizer-trajectory.npz') as z:before=z['before'];proposal_values=z['proposal'];after=z['after']
  for i,row in enumerate(records):
   old=before[i,17438:17445].copy();proposal=proposal_values[i,17438:17445].copy()
   if row['parameter_grad_is_none'][2]:expected=old
   else:
    delta=proposal-old;expected=old+delta*np.float32(min(1.,p['speed_weight_step_norm']/max(float(np.linalg.norm(delta)),1e-30)));expected*=np.float32(min(1.,p['speed_weight_total_norm']/max(float(np.linalg.norm(expected)),1e-30)));expected=np.clip(expected,-p['speed_weight_absolute_bound'],p['speed_weight_absolute_bound']);expected[-1]=min(expected[-1],0.);expected=project(expected,p);count+=1
   maximum=max(maximum,float(abs(expected-after[i,17438:17445]).max()))
  assert maximum<2e-7 and count==331;reports[arm]=dict(projections=count,updates=len(records),maximum_parameter_difference=maximum,direct_power_input_sensitivity_limit=p['maximum_power_input_sensitivity'])
 write_json(ROOT/'power-projection-audit-01.json',dict(status='completed_independent_fixture_adjoint_power_projection_audit',models=reports,complete_circuit_stability_certified=False,actor_qualified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
