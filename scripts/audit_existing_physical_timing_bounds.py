"""Apply interval arithmetic to existing non-fit candidates without inventing clocks."""
import argparse,json
from pathlib import Path
from collections import Counter
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.physical_timing_intervals import TimeBound,compare_physical_timing

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');sources=[]
 for role in ('train','development'):
  prefix=f'20261031-pace-{role}-native-copy-candidate-00';root=base/prefix/'recording';audit=read(base/(prefix+'-authority-audit')/'result.json');raw=read(base/(prefix+'-raw-audit')/'result.json');result=read(root/'result.json');events=root/'events.jsonl'
  if result.get('training_admissible') is not False or raw.get('continuous_40ms_training_admissible') is not False or audit['source_events_sha256']!=sha256(events) or audit.get('native_camera_source_time_synchronized') is not False or audit.get('exact_native_source_clock_verified') is not False:raise ValueError('Exact historical non-fit source required')
  counts=Counter();count=0
  with events.open(encoding='utf8') as f:
   for line in f:
    item=json.loads(line);when=item['decision_monotonic_seconds'];decision=TimeBound(when,when,'host_qpc_seconds',f'original-decision-{item["index"]}')
    # Render-call lower bounds, native copy receipts, and AI flags are not
    # certified bounds on physical exposure, sample, or control application.
    timing=compare_physical_timing(decision=decision,views=(None,)*4,body=None,action=None,maximum_view_age=.15,maximum_body_age=.05,minimum_action_delay=.015,maximum_action_delay=.065)
    if timing['training_admissible'] is not False:raise ValueError('Timing arithmetic cannot admit training')
    counts[timing['status']]+=1;count+=1
  if count!=audit['observations']:raise ValueError('Whole source count differs')
  sources.append(dict(role=role,decisions=count,events_sha256=sha256(events),authority_result_sha256=sha256(base/(prefix+'-authority-audit')/'result.json'),timing_status_counts=dict(counts),physical_exposure_bounds='unavailable',physical_body_sample_bounds='unavailable',physical_action_application_bounds='unavailable',fit_allowed=False))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),timing_source_sha256=sha256('src/flybrain_drive/physical_timing_intervals.py'),sources=sources,optimizer_updates=0))
 result=dict(status='completed_nonpromoting_interval_review',sources=sources,training_admissible=False,actor_deployed=False,vehicle_control_sent=False,interpretation='Explicit missing physical-time bounds, not evidence that host inputs were synchronous or asynchronous')
 write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
