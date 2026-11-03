"""Audit matched-body view-hold effects in one recurrent circuit."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('D:/CodexResearchRuns/flybrain-fixed-body-view-20261102-01');REF=Path('D:/CodexResearchRuns/flybrain-view-lesion-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-fixed-body-view-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-fixed-body-view-analysis-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads(PLAN.read_text());r=json.loads((ROOT/'result.json').read_text());rr=json.loads((REF/'result.json').read_text());trajectory=ROOT/'trajectory.npz';scene=ROOT/'scenes.npz';ref=REF/'trajectory.npz'
 if r['status']!='completed_fixed_body_synthetic_view_lesion' or r['actor_qualified'] is not False or sha(trajectory)!=r['output_sha256'] or sha(scene)!=r['scenes_sha256'] or sha(ref)!=p['reference_sha256'] or sha(ref)!=rr['output_sha256'] or r['plan_sha256']!=sha(PLAN):raise ValueError('completed pinned replay required')
 results=[];body_diff=0.;nonheld_diff=0.;baseline_out=0.;frozen_retina=0.
 with np.load(trajectory) as tr,np.load(scene) as views,np.load(ref) as old:
  for seed in p['seeds']:
   for bend in (-1,0,1):
    key=f'baseline_{seed}_{bend}';a=tr[key];old_a=old[key];baseline_out=max(baseline_out,float(np.max(np.abs(a[:,6:8]-old_a[:,6:8]))));scene_base=views[key].reshape(81,4,9,16)
    for mode,held in [('freeze_front',[0]),('freeze_sides',[1,2]),('freeze_rear',[3])]:
     k=f'{mode}_{seed}_{bend}';b=tr[k];sc=views[k].reshape(81,4,9,16);other=[x for x in range(4) if x not in held]
     if a.shape!=(81,16) or b.shape!=(81,16) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('bad paired history')
     body_diff=max(body_diff,float(np.max(np.abs(a[:,:6]-b[:,:6]))));nonheld_diff=max(nonheld_diff,float(np.max(np.abs(scene_base[:,other]-sc[:,other]))));frozen_retina=max(frozen_retina,float(np.max(np.abs(sc[:,held,0,:]-sc[0:1,held,0,:]))))
     for name,lo,hi in [('before',9,11.5),('after',13,16)]:
      mask=(a[:,1]>=lo)&(a[:,1]<hi)
      if not mask.any():raise ValueError('empty paired window')
      delta=b[mask,6:8]-a[mask,6:8];results.append({'seed':seed,'bend':bend,'lesion':mode,'window':name,'frames':int(mask.sum()),'mean_abs_turn_intent_change':float(np.mean(np.abs(delta[:,0]))),'max_abs_turn_intent_change':float(np.max(np.abs(delta[:,0]))),'mean_abs_forward_intent_change':float(np.mean(np.abs(delta[:,1])))})
 summary={}
 for mode in ('freeze_front','freeze_sides','freeze_rear'):
  summary[mode]={}
  for window in ('before','after'):
   rows=[x for x in results if x['lesion']==mode and x['window']==window];summary[mode][window]={'case_mean_abs_turn_range':[float(min(x['mean_abs_turn_intent_change'] for x in rows)),float(max(x['mean_abs_turn_intent_change'] for x in rows))],'case_mean_abs_turn_average':float(np.mean([x['mean_abs_turn_intent_change'] for x in rows])),'case_mean_abs_forward_max':float(max(x['mean_abs_forward_intent_change'] for x in rows))}
 report={'status':'completed_fixed_body_synthetic_view_analysis','plan_sha256':sha(PLAN),'source_result_sha256':sha(ROOT/'result.json'),'program_sha256':sha(Path(__file__)),'body_max_difference':body_diff,'unheld_scene_max_difference':nonheld_diff,'frozen_retina_max_difference':frozen_retina,'baseline_intent_replay_max_difference':baseline_out,'summary':summary,'results':results,'actor_qualified':False,'training_admissible':False,'physical_time_verified':False,'limits':'Fixed body history, independent recurrent resets; frozen view retains temporal FlyVis dynamics. Previously observed synthetic scenes, single road geometry, no biological or real-driving proof.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k not in ('results',)},indent=2))
if __name__=='__main__':main()
