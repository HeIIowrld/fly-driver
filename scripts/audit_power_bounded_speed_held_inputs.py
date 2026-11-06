"""Independent late-window variation arithmetic for held synthetic input stress."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 path=ROOT/'held-input-probe-01';r=read(path/'result.json');assert sha256(path/'values.npz')==r['values_sha256'];maximum=0.;reports=[]
 with np.load(path/'values.npz') as z:
  for j,row in enumerate(r['reports']):
   for field,reported in [('prediction','late_action_peak_to_peak'),('command','late_command_peak_to_peak'),('muscles','late_muscle_peak_to_peak')]:
    x=z[f'case{j}_{field}'];assert x.shape[0]==301 and np.isfinite(x).all();np.testing.assert_array_equal(np.ptp(x[-51:],axis=0),row[reported])
   x=z[f'case{j}_prediction'];maximum=max(maximum,float(np.ptp(x[-51:],axis=0).max()));np.testing.assert_array_equal(abs(x[-1]-x[-2]),row['last_step_action_delta']);reports.append(dict(arm=row['arm'],speed=row['speed_m_s'],temperature=row['temperature_c'],speed_policy_off=row['speed_policy_off'],late_action_peak_to_peak=row['late_action_peak_to_peak']))
 assert len(reports)==24;write_json(ROOT/'held-input-audit-01.json',dict(status='completed_independent_held_input_variation_audit',reports=reports,maximum_late_action_peak_to_peak=maximum,physical_stability_verified=False,actor_qualified=False,vehicle_control_sent=False));print(maximum)
if __name__=='__main__':main()
