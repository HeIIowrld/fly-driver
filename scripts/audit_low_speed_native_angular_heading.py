"""Offline-only comparison of native angular components and diagnostic absolute heading."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX,decode_physics_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
roots=list(map(Path,a.probe));run=begin_run(a.output,dict(program_sha256=sha256(__file__),sources=[dict(path=str(p),events_sha256=sha256(p/'events.jsonl'),result_sha256=sha256(p/'result.json')) for p in roots],world_heading='offline diagnostic only; prohibited actor input',seed=863009))
reports=[]
for root in roots:
 r=json.loads((root/'result.json').read_text());events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
 if r['status']!='completed_low_speed_pulse_diagnostic' or not r['post_stop_verified']:raise ValueError('Completed stopped diagnostic required')
 rows=[]
 for event in events:
  if event['kind']!='native':continue
  n=event['native'];raw=base64.b64decode(n['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=n['prefix_sha256']:raise ValueError('Native bytes changed')
  sample=decode_physics_prefix(raw,n['received_monotonic_seconds']);v=PHYSICS_PREFIX.unpack(raw);offset=0;head=None
  for name,_,count in PHYSICS_LAYOUT:
   if name=='heading':head=v[offset]
   offset+=count
  if sample.packet_id!=n['packet_id'] or head is None:raise ValueError('Native identity mismatch')
  rows.append((sample.received_monotonic_seconds,head,*sample.fields['localAngularVel']))
 rows=np.asarray(rows);dt=np.diff(rows[:,0]); dh=np.diff(np.unwrap(rows[:,1]));w=(rows[1:,2:]+rows[:-1,2:])/2
 valid=(dt>=.005)&(dt<=.15);predict=w[valid]*dt[valid,None];target=dh[valid]
 fits=[]
 for j in range(3):
  x=predict[:,j];den=float(x@x);slope=float(x@target/den) if den>0 else None
  fits.append(dict(component=j,least_squares_scale=slope,delta_sum=float(x.sum()),
   residual_rms=float(np.sqrt(np.mean((target-x)**2))), residual_rms_inverted=float(np.sqrt(np.mean((target+x)**2)))))
 reports.append(dict(probe=str(root),packets=len(rows),valid_intervals=int(valid.sum()),excluded_equal_or_short=int((dt<.005).sum()),
  excluded_long=int((dt>.15).sum()),elapsed_host_seconds=float(dt.sum()),heading_range_native=float(np.ptp(rows[:,1])),
  heading_delta_valid_sum=float(target.sum()),components=fits))
result=dict(status='completed_offline_heading_comparison',reports=reports,world_heading_actor_input=False,
 source_physical_time_verified=False,heading_physical_units_verified=False,physiological_axis_verified=False,
 training_allowed=False,actuator_messages_sent=0)
write_json(run/'result.json',result);print(json.dumps(result))
