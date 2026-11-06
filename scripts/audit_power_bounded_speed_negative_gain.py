"""Independent numerical arithmetic for deliberate sensitivity-violation controls."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 path=ROOT/'power-gain-negative-control-02';r=read(path/'result.json');assert sha256(path/'values.npz')==r['values_sha256'] and r['maximum_original_on_off_replay_error']<2e-6 and r['checkpoint_modified'] is False
 with np.load(path/'values.npz') as z:
  for j,row in enumerate(r['reports']):
   p=z[f'case{j}_prediction'];c=z[f'case{j}_command'];assert len(p)==len(c)==301 and np.isfinite(p).all() and np.isfinite(c).all();np.testing.assert_array_equal(np.ptp(p[-51:],axis=0),row['late_action_peak_to_peak']);np.testing.assert_array_equal(np.ptp(c[-51:],axis=0),row['late_command_peak_to_peak'])
 values={v['power_coefficient_multiplier']:max(v['late_action_peak_to_peak']) for v in r['reports']};assert values[0.]==values[1.]==0 and values[4.]>.1 and values[8.]>.1
 write_json(ROOT/'negative-power-gain-audit-01.json',dict(status='completed_independent_power_gain_negative_control',reports=r['reports'],maximum_original_replay_error=r['maximum_original_on_off_replay_error'],same_checkpoint_other_parameters_fixed=True,scope='One declared synthetic constant-input case; not physical stability or a promoted larger-gain policy.',actor_qualified=False,vehicle_control_sent=False));print(values)
if __name__=='__main__':main()
