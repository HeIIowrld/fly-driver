"""Count decision-time host freshness against exact fit-forbidden original records."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=B/'20261029-t1434-body21-flight-cache';manifest=root/'dataset.json'
 if sha256(manifest)!=read(root/'result.json')['dataset_manifest_sha256']:raise ValueError('dataset changed')
 ep=next(e for e in read(manifest)['episodes'] if e['file']=='episode-06.npz');record=Path(ep['original']['recording']);eventfile=record/'events.jsonl'
 if read(record/'result.json').get('training_admissible') is not False or sha256(eventfile)!=ep['source_events_sha256']:raise ValueError('fit forbidden original source required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(eventfile),body_limit=.05,view_limit=.15,scope='host decision-time only'))
 try:
  events=[json.loads(line) for line in eventfile.read_text(encoding='utf8').splitlines()];ages=[];held=[];viewages=[];previous=None
  if len(events)!=ep['observations']:raise ValueError('incomplete source')
  for i,e in enumerate(events):
   now=float(e['decision_monotonic_seconds']);receipt=float(e['native']['received_monotonic_seconds']);camera=e['camera'];views=camera['views'];received=float(camera['received'])
   if e['index']!=i or [v['view_id'] for v in views]!=['front','left','right','rear'] or not 0<=receipt<=now or not 0<=received<=now:raise ValueError('invalid original host order')
   lower=np.array([float(v['render_started_lower']) for v in views]);vage=now-lower
   if not np.isfinite(vage).all() or np.any(lower>received) or (lower<0).any():raise ValueError('invalid view bounds')
   ages.append(now-receipt);held.append(np.nan if previous is None else now-previous);viewages.append(vage);previous=receipt
  ages=np.array(ages);held=np.array(held);viewages=np.array(viewages);valid=np.arange(len(ages))>0
  current=(ages<=.05);prior=(held<=.05);views=(viewages.max(axis=1)<=.15);joint=valid&current&prior&views
  output=run/'ages.npz';np.savez_compressed(output,current=ages,held=held,views=viewages,joint=joint)
  result=dict(status='completed_historical_host_freshness_audit',decisions=len(ages),first_priming_excluded=1,current_stale=int((~current).sum()),held_stale_after_first=int((valid&~prior).sum()),view_stale=int((~views).sum()),all_pass_after_first=int(joint.sum()),eligible_after_first=int(valid.sum()),current_age_p95=float(np.quantile(ages,.95)),held_age_p95=float(np.quantile(held[valid],.95)),view_age_max_p95=float(np.quantile(viewages.max(axis=1),.95)),output_sha256=sha256(output),training_admissible=False,physical_exposure_sync_verified=False,output_time_freshness_checked=False,actor_deployed=False,vehicle_control_sent=False)
 except Exception as e:result=dict(status='failed',reason=str(e),error=type(e).__name__)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
