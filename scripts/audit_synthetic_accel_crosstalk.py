"""Posthoc cross-axis review of synthetic acceleration training results."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('D:/CodexResearchRuns/flybrain-synthetic-accel-fit-20261102-01');OUT=Path('D:/CodexResearchRuns/flybrain-synthetic-accel-crosstalk-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((ROOT/'result.json').read_text(encoding='utf8'))
 if r['status']!='completed_synthetic_acceleration_training' or sha(ROOT/'validation.npz')!=r['output_sha256'] or r['actor_qualified'] is not False:raise ValueError('completed unqualified run required')
 rows=[]
 with np.load(ROOT/'validation.npz',allow_pickle=False) as z:
  for k,seed in enumerate((879715,879716)):
   for bend in (1,2):
    for kind in ('forward','braking','left','right'):
     a=z[f'{k}_{bend}_{kind}'];mean=a.mean(axis=0);primary=1 if kind in ('forward','braking') else 0;other=1-primary;scales=[.003,.002]
     entry=next(x for x in r['validation'] if x['seed']==seed and x['bend']==bend and x['kind']==kind)
     if not np.allclose(mean,entry['mean_delta'],atol=1e-8):raise ValueError('saved validation differs')
     rows.append({'seed':seed,'bend':bend,'kind':kind,'primary_normalized':float(abs(mean[primary])/scales[primary]),'off_axis_normalized':float(abs(mean[other])/scales[other]),'mean_delta':mean.tolist()})
 long=[x for x in rows if x['kind'] in ('forward','braking')]
 report={'status':'completed_posthoc_cross_axis_audit','source_result_sha256':sha(ROOT/'result.json'),'program_sha256':sha(Path(__file__)),'predeclared_screen_passed':r['predeclared_screen_passed'],'predeclared_screen_did_not_bound_cross_axis':True,'longitudinal_min_off_axis_normalized':min(x['off_axis_normalized'] for x in long),'longitudinal_max_primary_normalized':max(x['primary_normalized'] for x in long),'candidate_qualified':False,'physical_time_verified':False,'training_admissible':False,'rows':rows,'limits':'Posthoc audit; steering/forward intent scales are hypothetical target scales, not physical units, biological or closed-loop quality.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
