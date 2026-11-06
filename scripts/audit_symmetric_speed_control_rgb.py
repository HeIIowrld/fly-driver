"""Audit reused original RGB feature provenance and newly replayed student branches."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 raw=Path('runs/spatial-motion-dn-learning-20261104/real-rgb-fork-02');rr=read(raw/'result.json');assert sha256(raw/'features.npz')==rr['features_sha256'] and sha256(raw/'branches.json')==rr['branches_sha256'];branches=read(raw/'branches.json');assert len({r['initial_full_visual_state_sha256'] for r in branches})==len({r['initial_held_retina_sha256'] for r in branches})==1
 path=ROOT/'neutral-real-rgb-student-fork-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];reports={}
 with np.load(path/'predictions.npz') as z:
  for mode,row in r['reports'].items():
   x=z[mode+'_prediction'];margin=float(-(x[1,:,0]-x[0,:,0]).mean()/2);assert abs(margin-row['signed_teacher_bend_margin'])<1e-12 and (margin>0)==row['direction_correct'] and (margin>=.002)==row['target_margin_passed'];assert float(abs(x[0]-x[2]).max())<2e-6;reports[mode]=dict(margin=margin,maximum_action_difference_by_axis=abs(x[1]-x[0]).max(0).tolist())
 assert r['prefix_replay_maximum_difference']==0 and r['identity_factual_prediction_maximum_difference']<2e-6
 write_json(ROOT/'neutral-real-rgb-audit-01.json',dict(status='completed_independent_speed_policy_rgb_fork_audit',source_features_sha256=rr['features_sha256'],reports=reports,groups=1,upstream_flyvis_state_rebranched=True,physical_counterfactual_verified=False,actor_qualified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
