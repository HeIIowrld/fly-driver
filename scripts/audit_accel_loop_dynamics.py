"""Compare paired synthetic closed-loop acceleration feedback at common distance."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('D:/CodexResearchRuns/flybrain-synthetic-accel-loop-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-accel-loop-dynamics-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-accel-loop-dynamics-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads(PLAN.read_text());r=json.loads((ROOT/'result.json').read_text());path=ROOT/'trajectory.npz'
 if r['status']!='completed_synthetic_acceleration_closed_loop' or sha(ROOT/'result.json')!=p['source_result_sha256'] or sha(path)!=p['source_trajectory_sha256'] or sha(path)!=r['output_sha256'] or r['actor_qualified'] is not False:raise ValueError('source changed or qualified')
 grid=np.linspace(8.5,15.5,71);out=[]
 with np.load(path,allow_pickle=False) as z:
  for speed in (3,6):
   for bend in (-1,0,1):
    a=z[f'{speed}_{bend}_base'];b=z[f'{speed}_{bend}_candidate']
    for t in (a,b):
     if t.ndim!=2 or t.shape[1]!=16 or not np.isfinite(t).all() or np.any(np.diff(t[:,1])<=0) or t[0,1]>grid[0] or t[-1,1]<grid[-1]:raise ValueError('complete strictly advancing paired trajectories required')
    def interp(t,col):return np.interp(grid,t[:,1],t[:,col])
    diff={name:interp(b,i)-interp(a,i) for name,i in [('forward_intent',6),('steer_rad',7),('acceleration_m_s2',8),('speed_m_s',4),('time_s',0)]}
    row={'speed':speed,'bend':bend,'forward_acceleration_input_abs_median_m_s2':float(np.median(np.abs(interp(b,9)))),'lateral_acceleration_input_abs_median_m_s2':float(np.median(np.abs(interp(b,10)))),'at_15_5m':{name:float(v[-1]) for name,v in diff.items()},'maximum_absolute_difference':{name:float(np.max(np.abs(v))) for name,v in diff.items()}}
    out.append(row)
 report={'status':'completed_posthoc_synthetic_dynamics_comparison','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'source_result_sha256':sha(ROOT/'result.json'),'conditions':out,'actor_qualified':False,'physical_time_verified':False,'training_admissible':False,'limits':p['limits']}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
