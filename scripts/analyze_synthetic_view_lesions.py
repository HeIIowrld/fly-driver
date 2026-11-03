"""Compare synthetic view-hold interventions at the same route distances."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('D:/CodexResearchRuns/flybrain-view-lesion-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-view-lesion-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-view-lesion-analysis-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads(PLAN.read_text());r=json.loads((ROOT/'result.json').read_text());path=ROOT/'trajectory.npz'
 if r['status']!='completed_synthetic_view_lesion_collection' or r['actor_qualified'] is not False or sha(path)!=r['output_sha256'] or r['plan_sha256']!=sha(PLAN):raise ValueError('completed pinned synthetic diagnostic required')
 rows=[]
 with np.load(path,allow_pickle=False) as z:
  for seed in p['seeds']:
   for bend in (-1,0,1):
    def measure(mode):
     a=z[f'{mode}_{seed}_{bend}']
     if a.shape!=(81,16) or not np.isfinite(a).all() or np.any(np.diff(a[:,1])<=0) or not a[0,1]<=10<16<=a[-1,1]:raise ValueError('complete monotonic history required')
     return {'signed_center_error_at_16m':float(np.interp(16,a[:,1],a[:,2])-bend*.018*(16-12)**2),'steer_at_10m_rad':float(np.interp(10,a[:,1],a[:,8]))}
    base=measure('baseline')
    for mode in ('freeze_front','freeze_sides','freeze_rear'):
     x=measure(mode);rows.append({'seed':seed,'bend':bend,'lesion':mode,'baseline':base,'intervention':x,'absolute_error_change_m':abs(x['signed_center_error_at_16m'])-abs(base['signed_center_error_at_16m']),'steer_change_at_10m_rad':x['steer_at_10m_rad']-base['steer_at_10m_rad']})
 summary={}
 for mode in ('freeze_front','freeze_sides','freeze_rear'):
  a=[x for x in rows if x['lesion']==mode];summary[mode]={'cases':len(a),'mean_absolute_error_change_m':float(np.mean([x['absolute_error_change_m'] for x in a])),'min_absolute_error_change_m':float(min(x['absolute_error_change_m'] for x in a)),'max_absolute_error_change_m':float(max(x['absolute_error_change_m'] for x in a)),'max_abs_steer_change_at_10m_rad':float(max(abs(x['steer_change_at_10m_rad']) for x in a))}
 report={'status':'completed_synthetic_view_lesion_analysis','plan_sha256':sha(PLAN),'source_result_sha256':sha(ROOT/'result.json'),'source_trajectory_sha256':sha(path),'program_sha256':sha(Path(__file__)),'summary':summary,'rows':rows,'actor_qualified':False,'physical_time_verified':False,'training_admissible':False,'limits':'View holding is an artificial closed-loop intervention; resulting car states diverge. Synthetic single road geometry, two textures, no physiological or real AC validation; reference line used only offline for metrics.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
