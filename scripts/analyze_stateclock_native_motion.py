"""Compare CSP cached speed and native copies around render, without asserting sync."""
import base64,bisect,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.diagnostic_cached_body_clock import summarize_cached_body_around_render
ROOT=Path('D:/CodexResearchRuns/flybrain-stateclock-motion-20261102-01');OUT=Path('D:/CodexResearchRuns/flybrain-stateclock-motion-analysis-20261102-01')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 r=json.loads((ROOT/'result.json').read_text(encoding='utf8'));c=json.loads((ROOT/'clock-after.json').read_text(encoding='utf8'));records=json.loads((ROOT/'native-copy-windows.json').read_text(encoding='utf8'));frames=json.loads((ROOT/'diagnostic-frames.json').read_text(encoding='utf8'))
 for file,key in [('native-copy-windows.json','native_copy_sha256'),('clock-after.json','clock_after_sha256'),('diagnostic-frames.json','metadata_sha256'),('live-frame-hashes.jsonl','host_observed_hashes_sha256')]:
  if sha(ROOT/file)!=r[key]:raise ValueError('changed source '+file)
 if r['status']!='completed' or len(frames)!=r['frames'] or len(records)!=r['native_copy_samples']:raise ValueError('incomplete source')
 host=[json.loads(x) for x in (ROOT/'live-frame-hashes.jsonl').read_text(encoding='utf8').splitlines()]
 if len(host)!=len(frames):raise ValueError('incomplete host observations')
 samples=[]
 for rec in records:
  a=rec['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('changed native prefix')
  time=float(rec['copy']['second_copy_end']);sample=decode_teacher_physics(raw,time)
  if sample.physics.packet_id!=a['packet_id']:raise ValueError('packet mismatch')
  samples.append((time,float(sample.physics.fields['speedKmh']),sample.physics.packet_id))
 times=[s[0] for s in samples]
 if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('nonmonotonic native copy')
 rows=[];mid=(c['offset_lower_seconds']+c['offset_upper_seconds'])/2
 for record,observed in zip(frames,host,strict=True):
  path=Path(record['source_file']);index=record['index']
  if observed['index']!=index or sha(path)!=record['sha256'] or sha(path)!=observed['metadata_sha256']:raise ValueError('frame digest mismatch')
  frame=json.loads(path.read_text(encoding='utf8'));summarize_cached_body_around_render(frame)
  if frame['request_id']!=r['request_id'] or frame['index']!=index or frame['generation']!=c['generation']:raise ValueError('frame identity mismatch')
  for view in frame['views']:
   img=observed['images'][view['view_id']]
   if img['file']!=view['file'] or sha(path.parent/img['file'])!=img['sha256']:raise ValueError('image changed')
   estimate=float(view['render_started_clock'])+mid;idx=bisect.bisect_left(times,estimate);candidates=[j for j in (idx-1,idx) if 0<=j<len(samples)]
   if not candidates:continue
   t,speed,pid=samples[min(candidates,key=lambda j:abs(times[j]-estimate))];csp=float(view['diagnostic_clock_pair_before']['body_diagnostic']['speed_kmh'])
   rows.append({'frame':index,'view':view['view_id'],'csp_speed_kmh':csp,'native_speed_kmh':speed,'native_packet':pid,'native_copy_minus_estimated_render_s':t-estimate,'speed_difference_kmh':speed-csp})
 moving=[x for x in rows if x['csp_speed_kmh']>1 and abs(x['native_copy_minus_estimated_render_s'])<=.05]
 if not moving:raise ValueError('no eligible moving observations')
 diffs=np.array([abs(x['speed_difference_kmh']) for x in moving]);dt=np.array([abs(x['native_copy_minus_estimated_render_s']) for x in moving]);OUT.mkdir(parents=True,exist_ok=False)
 report={'status':'completed_moving_cached_native_receipt_comparison','frames':len(frames),'view_pairs':len(rows),'eligible_moving_view_pairs':len(moving),'clock_offset_width_ms':(c['offset_upper_seconds']-c['offset_lower_seconds'])*1000,'maximum_csp_speed_kmh':max(x['csp_speed_kmh'] for x in rows),'moving_abs_speed_diff_kmh':{'median':float(np.median(diffs)),'p95':float(np.percentile(diffs,95)),'max':float(np.max(diffs))},'moving_abs_host_receipt_difference_ms':{'median':float(np.median(dt)*1000),'max':float(np.max(dt)*1000)},'program_sha256':sha(Path(__file__)),'source_result_sha256':sha(ROOT/'result.json'),'rows':rows,'physical_exposure_sync_verified':False,'native_packet_identity_verified':False,'training_admissible':False,'actor_deployed':False,'limitations':'nearest native host copy to midpoint of assumed-rate host render clock; neither is event time. Four views and frames correlated. Diagnostic future copy may be selected. No physical sync bound, axis validation, actor or fit.'}
 (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
