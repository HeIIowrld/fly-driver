"""Separate time-varying and mean effects in archived unified path ablations."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.source);r=json.loads((root/'result.json').read_text(encoding='utf8'));f=root/'ablations.npz'
 if r['status']!='completed_moving_teacher_shadow_path_ablations' or r['training_admissible'] is not False or sha256(f)!=r['output_sha256']:raise ValueError('Verified non-training source required')
 names=('visual_opponent','visual_spatial','visual_body_interaction','added_body_feedback');results={}
 with np.load(f,allow_pickle=False) as z:
  baseline=z['observed_intent']
  if baseline.shape!=(365,2) or not np.isfinite(baseline).all():raise ValueError('Exact finite original decisions required')
  for n in names:
   value=z[n+'_intent'];
   if value.shape!=baseline.shape or not np.isfinite(value).all():raise ValueError('Complete aligned interventions required')
   d=(value-baseline).astype(np.float64);parts={}
   for label,offset in (('all',0),('after_first_50',50)):
    x=d[offset:];parts[label]=dict(count=len(x),mean=x.mean(axis=0).tolist(),temporal_std=x.std(axis=0).tolist(),temporal_range=np.ptp(x,axis=0).tolist())
   results[n]=parts
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(root/'result.json'),source_arrays_sha256=sha256(f),posthoc_exclusion=50,optimizer_updates=0));out=dict(status='completed_posthoc_temporal_ablation_decomposition',conditions=results,training_admissible=False,vehicle_control_sent=False,actor_qualified=False,interpretation='One previously examined AI teacher trajectory; first-50 exclusion is post hoc, no route correctness or physiology inference.')
 write_json(run/'result.json',out);print(json.dumps(out,allow_nan=False))
if __name__=='__main__':main()
