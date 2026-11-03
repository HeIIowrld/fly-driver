"""Check preserved pooled retina and state continuity in view-hold runs."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('D:/CodexResearchRuns/flybrain-view-lesion-20261102-01');OUT=Path('D:/CodexResearchRuns/flybrain-view-lesion-integrity-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((ROOT/'result.json').read_text());sc=ROOT/'scenes.npz';tr=ROOT/'trajectory.npz'
 if r['status']!='completed_synthetic_view_lesion_collection' or sha(sc)!=r['scenes_sha256'] or sha(tr)!=r['output_sha256']:raise ValueError('source mismatch')
 diffs={};continuity=0.;initial=0.
 with np.load(sc,allow_pickle=False) as z,np.load(tr,allow_pickle=False) as a:
  for seed in (883901,883902):
   for bend in (-1,0,1):
    reference=z[f'baseline_{seed}_{bend}'].reshape(81,4,9,16)[0,:,0,:]
    for mode,views in [('freeze_front',[0]),('freeze_sides',[1,2]),('freeze_rear',[3])]:
     k=f'{mode}_{seed}_{bend}';s=z[k].reshape(81,4,9,16);initial=max(initial,float(np.max(np.abs(s[0,:,0,:]-reference))));diffs[k]=float(np.max(np.abs(s[:,views,0,:]-s[0:1,views,0,:])))
    for mode in ('baseline','freeze_front','freeze_sides','freeze_rear'):
     t=a[f'{mode}_{seed}_{bend}'];continuity=max(continuity,float(np.max(np.abs(t[:-1,10:14]-t[1:,1:5]))))
 report={'status':'completed_synthetic_view_hold_integrity','maximum_frozen_retina_difference':max(diffs.values()),'maximum_first_retina_difference':initial,'maximum_vehicle_state_continuity_difference':continuity,'source_result_sha256':sha(ROOT/'result.json'),'program_sha256':sha(Path(__file__)),'physical_time_verified':False,'actor_qualified':False,'training_admissible':False,'limits':'Recomputes stored outputs only, no independent neural rerun; frozen retina is an artificial intervention.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
