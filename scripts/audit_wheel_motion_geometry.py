"""Exploratory host-bracketed CSP/native wheel comparison; never actor input."""
import argparse,base64,bisect,hashlib,itertools,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.wheel_geometry_diagnostic import analyze_wheel_geometry

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.source);receipt=read(root/'result.json');run=begin_run(a.output,dict(source_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),diagnostic_only=True,actor=False));out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0)
 try:
  if receipt['status']!='completed' or sha256(root/'diagnostic-frames.json')!=receipt['metadata_sha256'] or sha256(root/'native-copy-windows.json')!=receipt['native_copy_sha256']:raise ValueError('exact completed originals required')
  frames=read(root/'diagnostic-frames.json');native=read(root/'native-copy-windows.json');clock=read(root/'clock-after.json');lo=float(clock['offset_lower_seconds']);hi=float(clock['offset_upper_seconds']);ends=[float(x['copy']['second_copy_end']) for x in native]
  if hi<lo or hi-lo>.1 or any(b<=a for a,b in zip(ends,ends[1:])):raise ValueError('ordered narrow host clock required')
  reports=[];missing=0;rejected=0
  for row in frames:
   path=Path(row['source_file'])
   if sha256(path)!=row['sha256']:raise ValueError('source frame digest mismatch')
   frame=read(path);geo=frame['wheel_geometry_diagnostic']
   if geo!=row['wheel_geometry'] or geo['generation']!=clock['generation']:raise ValueError('exact diagnostic generation required')
   try:g=analyze_wheel_geometry(geo)
   except ValueError:rejected+=1;continue
   host_lower=float(geo['started_source_clock'])+lo;host_upper=float(geo['finished_source_clock'])+hi;i=bisect.bisect_right(ends,host_lower)-1
   if i<0 or host_upper-ends[i]>.05:missing+=1;continue
   source=native[i]['authority'];raw=base64.b64decode(source['prefix_base64'],validate=True)
   if hashlib.sha256(raw).hexdigest()!=source['prefix_sha256']:raise ValueError('native packet hash mismatch')
   p=decode_teacher_physics(raw,ends[i]).physics
   if p.packet_id!=source['packet_id']:raise ValueError('native packet ID mismatch')
   csp=[float(w['load_candidate']) for w in geo['wheels']];load=[float(x) for x in p.fields['wheelLoad']];reports.append(dict(index=row['index'],labels=g['observed_labels'],outside_alignment=g['outside_side_alignment'],speed_m_s=float(p.fields['speedKmh'])/3.6,native_packet=p.packet_id,max_host_age_ms=(host_upper-ends[i])*1000,csp_loads=csp,native_loads=load))
  if len(reports)<20:raise ValueError('insufficient host-matched valid frames')
  A=np.array([x['csp_loads'] for x in reports]);B=np.array([x['native_loads'] for x in reports]);scores=[]
  for perm in itertools.permutations(range(4)):
   delta=np.abs(A-B[:,perm]);scores.append(dict(native_indices=list(perm),mean_abs=float(delta.mean()),median_abs=float(np.median(delta)),p95_abs=float(np.percentile(delta,95)),max_abs=float(delta.max())))
  scores.sort(key=lambda x:x['mean_abs']);speeds=np.array([x['speed_m_s'] for x in reports]);fast=speeds>=1.;labels_ok=sum(x['labels']==['FL','FR','RL','RR'] for x in reports)
  out.update(status='completed_exploratory_wheel_motion_geometry',frames_total=len(frames),matched=len(reports),missing=missing,geometry_rejected=rejected,geometry_identity_count=labels_ok,speed_min=float(speeds.min()),speed_max=float(speeds.max()),moving_frames_1ms=int(fast.sum()),max_host_age_ms=max(x['max_host_age_ms'] for x in reports),best=scores[0],second=scores[1],permutations=scores,rows=reports,physical_tick_verified=False,native_order_registered=False,training_admissible=False,interpretation='Posthoc host-clock bracket of preceding native copy; a frame and native copy may represent different physics ticks. Same car, short built-in-AI motion. CSP geometry and its relationship to native arrays are not independent measurement. No actor admission.')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',out)
 print(json.dumps({k:v for k,v in out.items() if k not in ('rows','permutations')},allow_nan=False))
 if out['status']!='completed_exploratory_wheel_motion_geometry':raise SystemExit(1)
if __name__=='__main__':main()
