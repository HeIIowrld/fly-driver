"""Independent numerical audit of held-input speed gain/channel interventions."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 reports={}
 for name in ['speed-gain-probe-01','speed-channel-feedback-probe-01']:
  path=ROOT/name;r=read(path/'result.json');assert r['status'].startswith('completed') and sha256(path/'values.npz')==r['values_sha256'] and r['maximum_original_on_off_replay_error']<2e-6 and r['checkpoint_modified'] is False
  with np.load(path/'values.npz') as z:
   for j,row in enumerate(r['reports']):
    p=z[f'case{j}_prediction'];c=z[f'case{j}_command'];assert len(p)==len(c)==301 and np.isfinite(p).all() and np.isfinite(c).all();np.testing.assert_array_equal(np.ptp(p[-51:],axis=0),row['late_action_peak_to_peak']);np.testing.assert_array_equal(np.ptp(c[-51:],axis=0),row['late_command_peak_to_peak']);assert float(c[-1,1]-row['speed_m_s'])==row['final_speed_error']
  reports[name]=r['reports']
 # This is the intervention result, not a physical-stability acceptance threshold.
 gain=reports['speed-gain-probe-01'];channel=reports['speed-channel-feedback-probe-01'];assert all(max(r['late_action_peak_to_peak'])==0 for r in gain if r['correction_gain']<=.5);assert all(max(r['late_action_peak_to_peak'])>.1 for r in gain if r['correction_gain']==1);assert max(next(r for r in channel if r['feature_ablation']=='power_pair_off')['late_action_peak_to_peak'])==0 and max(next(r for r in channel if r['feature_ablation']=='steering_pairs_off')['late_action_peak_to_peak'])>.1
 write_json(ROOT/'loop-intervention-audit-01.json',dict(status='completed_independent_speed_feedback_loop_interventions',reports=reports,gain_scope='Two fixed synthetic speed/temperature cases',channel_scope='One fixed synthetic speed/temperature case',actor_qualified=False,physical_stability_verified=False,vehicle_control_sent=False));print('Original replay parity, late variations and loop interventions verified.')
if __name__=='__main__':main()
