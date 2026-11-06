"""Independent RGB fork/baseline-feedback arithmetic and resume receipts."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/spatial-motion-dn-learning-20261104')
def main():
 raw=ROOT/'real-rgb-fork-02';rr=read(raw/'result.json');branches=read(raw/'branches.json');assert rr['upstream_flyvis_state_rebranched'] and sha256(raw/'branches.json')==rr['branches_sha256'] and sha256(raw/'features.npz')==rr['features_sha256'];assert len({x['initial_full_visual_state_sha256'] for x in branches})==1 and len({x['initial_held_retina_sha256'] for x in branches})==1
 path=ROOT/'real-rgb-student-fork-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];margins={};axes={}
 with np.load(path/'predictions.npz') as z:
  for mode,row in r['reports'].items():
   x=z[mode+'_prediction'];margin=float(-(x[1,:,0]-x[0,:,0]).mean()/2);assert abs(margin-row['signed_teacher_bend_margin'])<1e-12 and (margin>0)==row['direction_correct'];margins[mode]=margin;axes[mode]=np.max(abs(x[1]-x[0]),axis=0).tolist();assert float(abs(x[0]-x[2]).max())<2e-6
 source=ROOT/'original-fine-feedback-01';old=read(source/'result.json');assert sha256(source/'predictions.npz')==old['predictions_sha256'];counts=dict(total=0,yaw_damping=0,speed_damping=0,yaw_antidamping=0,speed_antidamping=0)
 with np.load(source/'predictions.npz') as z:
  for j,row in enumerate(old['reports']):
   counts['total']+=1
   for field in ['yaw','speed']:
    axis=lambda x:x[:,0] if field=='yaw' else x[:,1]-x[:,2];neutral=axis(z[f'case{j}_neutral_prediction']);minus=axis(z[f'case{j}_{field}_minus_prediction'])-neutral;plus=axis(z[f'case{j}_{field}_plus_prediction'])-neutral;sign=1 if field=='yaw' else -1;damp=bool(sign*minus.mean()<0 and sign*plus.mean()>0);anti=bool(sign*minus.mean()>0 and sign*plus.mean()<0);assert damp==row['feedback'][field]['mean_both_damping'] and anti==row['feedback'][field]['mean_both_antidamping'];counts[field+'_damping']+=int(damp);counts[field+'_antidamping']+=int(anti)
 assert counts==old['counts'];resume=read(ROOT/'optimizer-resume-receipt-01.json')
 for arm,row in resume['models'].items():
  f=ROOT/f'{arm}-fit-01/optimizer-resume-01/adam-state.pt';assert sha256(f)==row['adam_state_sha256'] and row['maximum_recorded_proposal_difference']==0;state=torch.load(f,map_location='cpu',weights_only=True);assert [int(v['step']) for v in state['state'].values()]==row['steps'];assert all(torch.isfinite(v[name]).all() for v in state['state'].values() for name in ['exp_avg','exp_avg_sq'])
 power_path=ROOT/'dedicated-power-route-01';power=read(power_path/'result.json');assert sha256(power_path/'values.npz')==power['values_sha256']
 with np.load(power_path/'values.npz') as z:assert float(abs(z['speed_error']).max())==0 and float(np.ptp(z['power'],axis=0).max())==power['dedicated_power_variation']
 write_json(ROOT/'extra-evidence-audit-01.json',dict(status='completed_independent_motion_dn_extra_evidence_audit',raw_rgb_signed_margins=margins,raw_rgb_maximum_action_differences_by_axis=axes,original_fine_fixed_target_body_feedback=counts,optimizer_moments_verified=True,dedicated_power=power,failed_rgb_attempt_preserved=read(ROOT/'real-rgb-fork-01/result.json')['status'],actor_qualified=False,physical_counterfactual_verified=False,actual_driving_improvement_verified=False));print(counts,margins)
if __name__=='__main__':main()
