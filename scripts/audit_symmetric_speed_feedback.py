"""Independent body-feedback signs with fixed versus naturally recomputed command."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 path=ROOT/'body-feedback-probe-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];counts={mode:dict(total=0,yaw=0,speed=0,nontrivial_yaw=0,nontrivial_speed=0) for mode in ['fixed_reference','actor_policy']};maximum=0.
 with np.load(path/'predictions.npz') as z,np.load(ROOT/'speed_policy-evaluation-01/predictions.npz') as factual:
  for row in r['reports']:
   t=row['task'];j=read('runs/multicorner-command-20261104/protocol-01.json')['tasks'].index(t);mode=row['mode'];prefix=f'case{j}_{mode}_';neutral=z[prefix+'neutral_prediction'];maximum=max(maximum,float(abs(neutral-factual[f"prediction_{t['episode']}"][t['begin']:t['end']]).max()));counts[mode]['total']+=1
   for field in ['yaw','speed']:
    axis=lambda x:x[:,0] if field=='yaw' else x[:,1]-x[:,2];minus=axis(z[prefix+field+'_minus_prediction'])-axis(neutral);plus=axis(z[prefix+field+'_plus_prediction'])-axis(neutral);sign=1 if field=='yaw' else -1;good=bool(sign*minus.mean()<0 and sign*plus.mean()>0);large=bool(good and abs(minus.mean())>1e-5 and abs(plus.mean())>1e-5);a=row['feedback'][field];assert good==a['mean_both_damping'] and large==a['mean_damping_above1e_minus5'];assert abs(minus.mean()-a['mean_minus_change'])<1e-12 and abs(plus.mean()-a['mean_plus_change'])<1e-12;counts[mode][field]+=int(good);counts[mode]['nontrivial_'+field]+=int(large)
 assert counts==r['counts'] and maximum==r['maximum_neutral_replay_difference'];write_json(ROOT/'body-feedback-audit-01.json',dict(status='completed_independent_speed_policy_body_feedback_audit',counts=counts,maximum_neutral_replay_difference=maximum,physical_feedback_stability_verified=False,actor_qualified=False,vehicle_control_sent=False));print(counts)
if __name__=='__main__':main()
