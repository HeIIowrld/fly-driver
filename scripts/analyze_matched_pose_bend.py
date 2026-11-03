"""Matched-body synthetic counterfactual analysis, no actor or vehicle control."""
import hashlib,json
from pathlib import Path
import numpy as np
RUN=Path('D:/CodexResearchRuns/flybrain-matched-pose-replay-20261102-01');REF=Path('D:/CodexResearchRuns/flybrain-dynamic-holdout-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-matched-pose-analysis-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-matched-pose-analysis-20261102-01')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def checked(root,status):
 r=json.loads((root/'result.json').read_text(encoding='utf-8'))
 if r['status']!=status:raise ValueError('source incomplete')
 for name,key in [('scenes.npz','scenes_sha256'),('trajectory.npz','output_sha256')]:
  if sha(root/name)!=r[key]:raise ValueError('source mismatch')
 return r
def main():
 checked(RUN,'completed_synthetic_matched_pose_replay');checked(REF,'completed_synthetic_dynamic_history_collection');plan=json.loads(PLAN.read_text(encoding='utf8'));OUT.mkdir(parents=True,exist_ok=False)
 cases={};reproduction={};match=0.
 with np.load(RUN/'trajectory.npz',allow_pickle=False) as tr,np.load(RUN/'scenes.npz',allow_pickle=False) as scenes,np.load(REF/'trajectory.npz',allow_pickle=False) as old,np.load(REF/'scenes.npz',allow_pickle=False) as oldsc:
  for seed in (883601,883602):
   key=f'baseline_{seed}_0';reproduction[key]={'scene_max_abs_diff':float(np.max(np.abs(scenes[key]-oldsc[key]))),'turn_max_abs_diff':float(np.max(np.abs(tr[key][:,6]-old[key][:,6])))}
   for mode in ('baseline','candidate'):
    base=tr[f'{mode}_{seed}_0'];mask=(base[:,1]>=9)&(base[:,1]<11.5)
    if not mask.any():raise ValueError('empty window')
    for bend in (-1,1):
     k=f'{mode}_{seed}_{bend}';a=tr[k]
     # Time, pose, speed, body yaw must be exactly paired.
     delta=float(np.max(np.abs(a[:,:6]-base[:,:6])));match=max(match,delta)
     if delta>1e-12:raise ValueError('body history mismatch')
     steer=a[mask,8]-base[mask,8]
     cases[k]={'frames':int(mask.sum()),'mean_steer_delta_rad':float(np.mean(steer)),'max_abs_steer_delta_rad':float(np.max(np.abs(steer))),'expected_sign_fraction':float(np.mean(steer*bend>0))}
 report={'status':'completed_synthetic_matched_pose_analysis','plan_sha256':sha(PLAN),'program_sha256':sha(__file__),'source_sha256':{str(p):sha(p) for p in [RUN/'result.json',RUN/'trajectory.npz',RUN/'scenes.npz',REF/'trajectory.npz',REF/'scenes.npz']},'max_matched_body_difference':match,'reference_reproduction':reproduction,'cases':cases,'physical_sync_verified':False,'actor_qualified':False,'vehicle_control_sent':False,'limits':plan['limitations']}
 (OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
