"""Audit physics marker evidence; diagnostic only, no actor input."""
import argparse,json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.source);receipt=read(root/'result.json');plan=read(a.plan);run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_sha256=sha256(root/'result.json'),plan_sha256=sha256(a.plan),actor=False));out=dict(status='failed',actor_deployed=False,optimizer_updates=0,vehicle_control_sent=False)
 try:
  path=root/'diagnostic-frames.json'
  if receipt['status']!='completed' or sha256(path)!=receipt['metadata_sha256']:raise ValueError('completed and exact source required')
  first={};seen={}
  for index,row in enumerate(read(path)):
   f=Path(row['source_file'])
   if index!=row['index'] or sha256(f)!=row['sha256']:raise ValueError('frame identity mismatch')
   raw=read(f);diag=raw['physics_marker_diagnostic'];wheel=raw['wheel_geometry_diagnostic']
   if diag!=row['physics_markers'] or wheel!=row['wheel_geometry'] or diag['actor_admissible'] is not False or diag['physical_sync_verified'] is not False:raise ValueError('exact diagnostic isolation required')
   for event in diag['events']:
    k=event['marker_index']
    if not 1<=k<=len(plan['markers']) or abs(event['progress']-plan['markers'][k-1])>1e-12:raise ValueError('unplanned marker')
    if k in seen and seen[k]!=event:raise ValueError('event changed between frames')
    seen[k]=event
    if k not in first:first[k]={'marker_index':k,'progress':event['progress'],'physics_event_ms':event['physics_event_ms'],'callback_sim_minus_event_ms':event['callback_sim_time_ms']-event['physics_event_ms'],'first_frame_index':index,'callback_to_first_geometry_ms':(wheel['started_source_clock']-event['callback_source_clock'])*1000}
  events=[first[k] for k in sorted(first)]
  if not events:raise ValueError('no marker observed')
  if any(b['physics_event_ms']<=a['physics_event_ms'] for a,b in zip(events,events[1:])):raise ValueError('marker event order differs')
  out.update(status='completed_physics_marker_audit',planned_markers=len(plan['markers']),observed_markers=len(events),events=events,native_tick_matched=False,camera_exposure_verified=False,training_admissible=False,interpretation='Physics-origin interpolated track events from CSP; callbacks observed on graphics context. Negative callback sim-minus-event values mean this comparison cannot be assumed causal latency. Track progress and event timestamps are diagnostics only, not actor inputs.')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',out)
 print(json.dumps(out,allow_nan=False))
 if out['status']!='completed_physics_marker_audit':raise SystemExit(1)
if __name__=='__main__':main()
