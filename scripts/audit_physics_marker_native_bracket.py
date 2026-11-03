"""Exploratory graphics interpolation and native-copy bracketing around physics events."""
import argparse,bisect,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=read(a.plan);root=Path(plan['source']);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),diagnostic_only=True,actor=False));out=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  for rel,expected in plan['source_sha256'].items():
   if sha256(root/rel)!=expected:raise ValueError('source digest mismatch: '+rel)
  receipt=read(root/'result.json');control=read(root/'control/result.json');api=read(root/'control/api-receipt.json');frames=read(root/'diagnostic-frames.json');copies=read(root/'native-copy-windows.json');bounds=read(root/'clock-after.json');marker=read(plan['marker_analysis'])
  if receipt['status']!='completed' or control['status']!='completed' or marker['status']!='completed_physics_marker_audit' or sha256(plan['marker_analysis'])!=plan['marker_analysis_sha256']:raise ValueError('original completed runs required')
  frame_clock=[];sim=[]
  for i,row in enumerate(frames):
   path=Path(row['source_file'])
   if row['index']!=i or sha256(path)!=row['sha256']:raise ValueError('exact source frame required')
   raw=read(path);g=raw['wheel_geometry_diagnostic']
   if g!=row['wheel_geometry'] or g['actor_admissible'] is not False:raise ValueError('original diagnostic required')
   frame_clock.append(float(g['started_source_clock']));sim.append(float(g['sim_time_before_ms']))
  frame_clock=np.array(frame_clock);sim=np.array(sim)
  if np.any(np.diff(frame_clock)<=0) or np.any(np.diff(sim)<=0):raise ValueError('ordered graphics timestamps required')
  telemetry=api['telemetry'];tele=[]
  for x in telemetry:
   diag=x.get('diagnostic_route')
   if not isinstance(diag,dict) or diag.get('actor_input') is not False:raise ValueError('diagnostic-only route required')
   tele.append((float(x['script_clock']),float(diag['spline_position'])))
  if any(b[0]<=a[0] for a,b in zip(tele,tele[1:])):raise ValueError('ordered telemetry required')
  starts=[float(x['copy']['first_copy_start']) for x in copies];ends=[float(x['copy']['second_copy_end']) for x in copies];lo=float(bounds['offset_lower_seconds']);hi=float(bounds['offset_upper_seconds'])
  if hi<lo or hi-lo>.1 or any(b<=a for a,b in zip(starts,starts[1:])) or any(b<=a for a,b in zip(ends,ends[1:])):raise ValueError('ordered bounded host clock required')
  events=[]
  for e in marker['events']:
   target=e['progress'];hits=[(a,b) for a,b in zip(tele,tele[1:]) if a[1]<=target<=b[1] and b[1]>a[1]]
   if len(hits)!=1:raise ValueError('unique graphics progress crossing required')
   left,right=hits[0];fraction=(target-left[1])/(right[1]-left[1]);cross=left[0]+fraction*(right[0]-left[0]);event_ms=e['physics_event_ms']
   if not sim[0]<=event_ms<=sim[-1] or not frame_clock[0]<=cross<=frame_clock[-1]:raise ValueError('event outside graphics sample range')
   predicted_ms=float(np.interp(cross,frame_clock,sim));event_clock=float(np.interp(event_ms,sim,frame_clock));lower=event_clock+lo;upper=event_clock+hi;earlier=bisect.bisect_right(ends,lower)-1;later=bisect.bisect_left(starts,upper)
   if earlier<0 or later>=len(copies):raise ValueError('native host bracket missing')
   events.append(dict(marker_index=e['marker_index'],physics_event_ms=event_ms,graphics_progress_interpolated_sim_ms=predicted_ms,graphics_minus_physics_ms=predicted_ms-event_ms,graphics_progress_sample_gap_ms=(right[0]-left[0])*1000,interpolated_event_source_clock=event_clock,estimated_host_interval_width_ms=(hi-lo)*1000,native_before_packet=copies[earlier]['authority']['packet_id'],native_after_packet=copies[later]['authority']['packet_id'],native_copy_span_ms=(starts[later]-ends[earlier])*1000))
  out.update(status='completed_exploratory_marker_native_bracket',events=events,physical_tick_verified=False,camera_exposure_verified=False,training_admissible=False,interpretation='Posthoc linear interpolation of graphics-rate sim time and progress around physics events; inferred host interval is conditional on interpolation and clock assumptions, not a proven bound on physical sampling or exposure. Native packet IDs bracket host copies only; track progress is diagnostic, never actor input.')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',out)
 print(json.dumps(out,allow_nan=False))
 if out['status']!='completed_exploratory_marker_native_bracket':raise SystemExit(1)
if __name__=='__main__':main()
