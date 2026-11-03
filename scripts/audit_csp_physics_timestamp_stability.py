"""Compare retained CSP physics timestamp progression and source clock only."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=read(a.plan);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),actor=False,diagnostic_only=True));out=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  reports=[]
  for item in plan['sources']:
   root=Path(item['root']);rr=read(root/'result.json');data=root/'diagnostic-frames.json'
   if rr['status']!='completed' or sha256(root/'result.json')!=item['result_sha256'] or sha256(data)!=item['frames_sha256'] or sha256(data)!=rr['metadata_sha256']:raise ValueError('source receipt mismatch')
   rows=read(data);source=[];physics=[]
   for i,row in enumerate(rows):
    path=Path(row['source_file'])
    if sha256(path)!=row['sha256']:raise ValueError('frame hash mismatch')
    d=read(path)['wheel_geometry_diagnostic']
    if row['index']!=i or d!=row['wheel_geometry'] or d['actor_admissible'] is not False:raise ValueError('exact diagnostic frame required')
    source.append(float(d['started_source_clock']));physics.append(float(d['car_physics_timestamp_ms'])/1000.)
   x=np.asarray(source);y=np.asarray(physics)
   if len(x)<20 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0):raise ValueError('ordered finite clocks required')
   delta=np.diff((x-y)*1000);relative=(x-y)*1000;reports.append(dict(name=item['name'],frames=len(x),source_interval_s=float(x[-1]-x[0]),physics_strictly_increasing=bool(np.all(np.diff(y)>0)),physics_minus_source_slope=float(np.polyfit(x-x[0],y-y[0],1)[0]),relative_offset_span_ms=float(np.ptp(relative)),adjacent_offset_change_abs_ms_p50=float(np.percentile(np.abs(delta),50)),adjacent_offset_change_abs_ms_p95=float(np.percentile(np.abs(delta),95)),adjacent_offset_change_abs_ms_max=float(np.max(np.abs(delta)))))
  out.update(status='completed_physics_timestamp_progression_diagnostic',reports=reports,physical_tick_verified=False,training_admissible=False,interpretation='Car timestamp is documented by installed CSP SDK as last physics record time, but native packets have no verified matching tick. Differences between source and physics timestamps include sampling and clock uncertainty, not a validated sensor latency. Same car/track, two short captures.')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',out)
 print(json.dumps(out,allow_nan=False))
 if out['status']!='completed_physics_timestamp_progression_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
