"""Posthoc lag sensitivity of preserved CSP/native wheel signals, not source-time registration."""
import argparse,base64,bisect,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=read(a.plan);root=Path(plan['source']);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),diagnostic_only=True,actor=False));r=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  receipt=read(root/'result.json');f=root/'diagnostic-frames.json';n=root/'native-copy-windows.json'
  if receipt['status']!='completed' or sha256(root/'result.json')!=plan['result_sha256'] or sha256(f)!=plan['frames_sha256'] or sha256(n)!=plan['native_sha256'] or sha256(f)!=receipt['metadata_sha256'] or sha256(n)!=receipt['native_copy_sha256']:raise ValueError('exact retained source required')
  bounds=read(root/'clock-after.json');mid=(bounds['offset_lower_seconds']+bounds['offset_upper_seconds'])/2;raw=read(n);native=[]
  for item in raw:
   a0=item['authority'];data=base64.b64decode(a0['prefix_base64'],validate=True)
   if hashlib.sha256(data).hexdigest()!=a0['prefix_sha256']:raise ValueError('native hash mismatch')
   t=float(item['copy']['second_copy_end']);s=decode_teacher_physics(data,t).physics
   if s.packet_id!=a0['packet_id']:raise ValueError('packet mismatch')
   native.append((t,s.fields['speedKmh']/3.6,np.asarray(s.fields['wheelLoad'],dtype=float)))
  times=[x[0] for x in native]
  if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('copy chronology required')
  frames=[]
  for x in read(f):
   path=Path(x['source_file'])
   if sha256(path)!=x['sha256']:raise ValueError('frame hash mismatch')
   g=read(path)['wheel_geometry_diagnostic']
   if g!=x['wheel_geometry'] or g['actor_admissible'] is not False or g['generation']!=bounds['generation']:raise ValueError('wheel geometry provenance')
   t=(g['started_source_clock']+g['finished_source_clock'])/2+mid;j=bisect.bisect_left(times,t);j=min(max(j,1),len(times)-1);j=j-1 if abs(times[j-1]-t)<=abs(times[j]-t) else j
   if native[j][1]>=plan['minimum_speed_m_s']:frames.append((x['index'],t,np.array([w['load_candidate'] for w in g['wheels']],dtype=float)))
  if len(frames)<plan['minimum_moving_frames']:raise ValueError('insufficient moving frames')
  split=len(frames)//2;results=[]
  for ms in plan['lags_ms']:
   errors=[];gaps=[]
   for index,t,csp in frames:
    target=t+ms/1000;j=bisect.bisect_left(times,target)
    if j<=0 or j>=len(times):raise ValueError('native range does not cover lag')
    j=j-1 if abs(times[j-1]-target)<=abs(times[j]-target) else j;gap=abs(times[j]-target)
    if gap>plan['max_nearest_host_gap_s']:raise ValueError('native sample gap too large')
    errors.append(float(np.mean(np.abs(csp-native[j][2]))));gaps.append(gap)
   errors=np.array(errors);results.append(dict(lag_ms=ms,first_mean=float(errors[:split].mean()),second_mean=float(errors[split:].mean()),first_median=float(np.median(errors[:split])),second_median=float(np.median(errors[split:])),max_host_gap_ms=float(max(gaps)*1000)))
  selected=min(results,key=lambda v:v['first_mean']);zero=next(x for x in results if x['lag_ms']==0);out=dict(status='completed_exploratory_wheel_host_lag',moving_frames=len(frames),first_frames=split,second_frames=len(frames)-split,chosen_lag_from_first_ms=selected['lag_ms'],first_mean=selected['first_mean'],held_second_mean=selected['second_mean'],zero_first_mean=zero['first_mean'],zero_second_mean=zero['second_mean'],all_lags=results,source_result_sha256=sha256(root/'result.json'),optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False,physical_tick_verified=False,native_order_registered=False,training_admissible=False,interpretation='Posthoc exploratory lag fit on one short autocorrelated capture. Future copies used for some candidate lags; not causal actor input. Host clocks do not establish source sampling or physics-tick sync. First/second split not independent.')
  r.update(out)
 except Exception as e:r.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',r)
 print(json.dumps(r,allow_nan=False))
 if r['status']!='completed_exploratory_wheel_host_lag':raise SystemExit(1)
if __name__=='__main__':main()
