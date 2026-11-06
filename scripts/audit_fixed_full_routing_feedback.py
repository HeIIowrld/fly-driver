"""Independent sensed-body damping arithmetic, distinct from command injection."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/fixed-full-routing-learning-20261104')
def main():
 path=ROOT/'body-feedback-probe-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];counts={role:dict(total=0,yaw=0,speed=0) for role in ['TRAIN','development']};nontrivial=dict(yaw=0,speed=0);maximum_neutral=0.
 with np.load(path/'predictions.npz') as z,np.load(ROOT/'visual_native-evaluation-01/predictions.npz') as original:
  for j,row in enumerate(r['reports']):
   task=row['task'];episode=task['episode'];role='TRAIN' if episode<2 else 'development';counts[role]['total']+=1;neutral=z[f'case{j}_neutral_prediction'];maximum_neutral=max(maximum_neutral,float(abs(neutral-original[f'prediction_{episode}'][task['begin']:task['end']]).max()))
   for field in ['yaw','speed']:
    v=lambda x:x[:,0] if field=='yaw' else x[:,1]-x[:,2];minus=v(z[f'case{j}_{field}_minus_prediction'])-v(neutral);plus=v(z[f'case{j}_{field}_plus_prediction'])-v(neutral);sign=1 if field=='yaw' else -1;correct=bool(sign*minus.mean()<0 and sign*plus.mean()>0);large=bool(correct and abs(minus.mean())>1e-5 and abs(plus.mean())>1e-5);assert correct==row['feedback'][field]['mean_both_damping'] and large==row['feedback'][field]['mean_damping_and_each_change_above1e_minus5'];assert abs(minus.mean()-row['feedback'][field]['mean_minus_change'])<1e-12 and abs(plus.mean()-row['feedback'][field]['mean_plus_change'])<1e-12;counts[role][field]+=int(correct);nontrivial[field]+=int(large)
 assert counts==r['counts'] and maximum_neutral==r['maximum_neutral_replay_difference'];write_json(ROOT/'body-feedback-audit-01.json',dict(status='completed_independent_sensed_body_feedback_audit',counts=counts,nontrivial_correct_cases_above1e_minus5=nontrivial,maximum_neutral_replay_difference=maximum_neutral,actor_qualified=False,physical_feedback_stability_verified=False));print(counts,nontrivial)
if __name__=='__main__':main()
