"""Summarize paired side-view interventions and left/right interaction."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('D:/CodexResearchRuns/flybrain-side-view-path-20261102-02');OUT=Path('D:/CodexResearchRuns/flybrain-side-view-path-summary-20261102-01');TRAJ=Path('D:/CodexResearchRuns/flybrain-fixed-body-view-20261102-01/trajectory.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((ROOT/'result.json').read_text());p=ROOT/'intent.npz'
 if r['status']!='completed_synthetic_side_path_interventions' or sha(p)!=r['output_sha256'] or sha(TRAJ)!=r['source_trajectory_sha256']:raise ValueError('source mismatch')
 summary={};interaction=[]
 for mode in ('left_all','right_all','both_all','left_retina','left_motion','right_retina','right_motion'):
  summary[mode]={}
  for window in ('before','after'):
   rows=[x['mean_abs_turn'] for x in r['reports'] if x['mode']==mode and x['window']==window]
   if len(rows)!=6:raise ValueError('incomplete cases')
   summary[mode][window]={'case_mean':float(np.mean(rows)),'case_min':float(min(rows)),'case_max':float(max(rows))}
 with np.load(p,allow_pickle=False) as z,np.load(TRAJ,allow_pickle=False) as t:
  for seed in (883901,883902):
   for bend in (-1,0,1):
    prefix=f'{seed}_{bend}_';a=t[f'baseline_{seed}_{bend}'];base=z[prefix+'baseline'];both=z[prefix+'both_all'];left=z[prefix+'left_all'];right=z[prefix+'right_all'];residual=both-left-right+base
    for name,lo,hi in [('before',9,11.5),('after',13,16)]:
     mask=(a[:,1]>=lo)&(a[:,1]<hi);interaction.append({'seed':seed,'bend':bend,'window':name,'mean_abs_turn_interaction':float(np.mean(np.abs(residual[mask,0])))})
 report={'status':'completed_side_path_summary','source_result_sha256':sha(ROOT/'result.json'),'program_sha256':sha(Path(__file__)),'summary':summary,'interaction':interaction,'maximum_mean_abs_turn_interaction':max(x['mean_abs_turn_interaction'] for x in interaction),'actor_qualified':False,'training_admissible':False,'limits':'Recalculation of stored output only; hybrid features can break natural neural relationships; results are correlating conditions on one synthetic course, not independent neural replication or biological performance.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='interaction'},indent=2))
if __name__=='__main__':main()
