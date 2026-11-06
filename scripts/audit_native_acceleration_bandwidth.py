"""Bandwidth-matched acceleration diagnostic; no actor observations are changed.

A least-squares velocity derivative over a window is a weighted average of
acceleration, not its instantaneous centre value. Compare equal temporal kernels
before interpreting a noisy vertical comparison as a coordinate problem.
"""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.diagnose_sac_anticipation import read
from scripts.audit_qualified_native_multisensory import native,acceleration_metrics
from scripts.sac_cohort_contract import DEFAULT_SPLIT,three_episode_cohort


def integrated_slope_weights(times):
 t=np.asarray(times,float)
 if t.ndim!=1 or len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Advancing sample times required')
 d=t-t.mean();c=d/(d@d);remaining=np.cumsum(c[::-1])[::-1][1:]
 half=.5*np.diff(t)*remaining;w=np.zeros(len(t));w[:-1]+=half;w[1:]+=half
 if not np.isclose(w.sum(),1.,atol=1e-8) or w.min()<-1e-8:raise ValueError('Invalid derivative-equivalent acceleration kernel')
 return w


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default=DEFAULT_SPLIT);a=p.parse_args();source=Path(a.audit);r=read(source/'result.json');entries=three_episode_cohort(a.split)
 run=begin_run(a.output,dict(arguments=vars(a),seed=780605,program_sha256=sha256(__file__),original_audit_sha256=sha256(source/'result.json'),comparison='Exact discrete LS derivative-equivalent trapezoidal acceleration kernel on original irregular native clocks',limitation='Body acceleration vectors averaged in their instantaneous local frames; central-frame approximation remains, no per-sample heading/attitude correction. Future samples are diagnostic only.',actor_deployed=False,vehicle_control_sent=False))
 out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False);reports=[]
 try:
  for i,entry in enumerate(entries):
   cfg=read(Path(entry['data'])/'manifest.json')['config'];root=Path(cfg['arguments']['recording']);cert=read(Path(cfg['arguments']['qualification'])/'result.json')
   ids,t,f,counts=native(root/'camera/native-body.jsonl',cert['native_body_sha256']);file=source/f'acceleration-diagnostic-{i}.npz';assert sha256(file)==r['files_sha256'][file.name]
   with np.load(file) as z:records=z['records']
   smoothed=[]
   for row in records:
    j=int(np.searchsorted(ids,int(row[1])));assert ids[j]==int(row[1]);lo=np.searchsorted(t,t[j]-.075);hi=np.searchsorted(t,t[j]+.075,side='right');weights=integrated_slope_weights(t[lo:hi]);smoothed.append(weights@(f['accG'][lo:hi][:,[2,0,1]]*9.80665))
   smoothed=np.asarray(smoothed);fast=records[:,3]>20;report=dict(episode=i,all_frames=acceleration_metrics(smoothed,records[:,7:10]),fast_frames=acceleration_metrics(smoothed[fast],records[fast,7:10]),original_fast=r['episodes'][i]['acceleration_fast'])
   reports.append(report);np.savez_compressed(run/f'matched-{i}.npz',smoothed_native_m_s2=smoothed,target_m_s2=records[:,7:10],fast=fast)
  out.update(status='completed_bandwidth_matched_acceleration_diagnostic',episodes=reports,original_assessment_or_masks_changed=False,acceleration_body_registered=False,actor_qualified=False,training_admissible=False,termination_reason='Diagnostic bandwidth comparison completed; no acceleration axis promoted or actor input changed')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit bandwidth audit failure')
 write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
