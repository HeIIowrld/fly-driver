"""Independent arithmetic for visual channel factorial trajectories."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read

ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 path=ROOT/'visual-channel-factorial-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];margins={f:[] for f in ['full','coarse_static','fine_motion','identity']}
 with np.load(path/'predictions.npz') as z:
  for row in r['reports']:
   for factor in margins:
    value=z[f"group{row['group']}_{factor}_prediction"];m=float(-(value[0,:,0]-value[1,:,0]).mean()/2);assert abs(m-row['factors'][factor]['signed_teacher_bend_margin'])<1e-12;margins[factor].append(m)
 for f,v in margins.items():assert sum(x>0 for x in v)==r['direction_counts'][f] and sum(x>=.002 for x in v)==r['target_counts'][f] and abs(float(np.mean(v))-r['mean_signed_margins'][f])<1e-12
 assert r['maximum_identity_left_right_difference']==0 and r['maximum_identity_replay_difference']<2e-6 and r['maximum_full_replay_difference']<2e-6
 write_json(ROOT/'visual-channel-audit-01.json',dict(status='completed_independent_visual_channel_arithmetic',direction_counts=r['direction_counts'],target_counts=r['target_counts'],mean_signed_margins=r['mean_signed_margins'],mean_nonlinear_interaction=float(np.mean(np.array(margins['full'])-np.array(margins['coarse_static'])-np.array(margins['fine_motion']))),groups=len(r['reports']),actor_qualified=False,actual_driving_improvement_verified=False))
if __name__=='__main__':main()
