"""Independently audit bounded moving steering diagnostic from original bytes; no control."""
import argparse,base64,hashlib,json,statistics
from pathlib import Path
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.probe_low_speed_steering_response import diagnostic

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',required=True);p.add_argument('--output',required=True);p.add_argument('--amplitude',required=True,type=float);a=p.parse_args()
root=Path(a.probe);outcome=json.loads((root/'result.json').read_text());events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_events_sha256=sha256(root/'events.jsonl'),source_result_sha256=sha256(root/'result.json'),actor=False))
if outcome['status']!='completed_low_speed_pulse_diagnostic' or not outcome['post_stop_verified'] or outcome['final_vjoy_status']!=1:raise ValueError('Successful stopped source required')
last=-1;last_clock=-1.;control=None; control_since=None;groups={'positive':[],'negative':[]};samples=0;maxspeed=0.;first=None;last_sample=None
for event in events:
 if event['kind'] in ('open','request'):
  req=event['request'];c=req['requested_control'];time=req['requested_monotonic_seconds']
  if not req['driver_accepted'] or c['steering'] not in (-a.amplitude,0.,a.amplitude) or c['throttle'] not in (0.,.15) or c['brake'] not in (0.,1.):raise ValueError('Unexpected request')
  if control is None or c['steering']!=control['steering']:control_since=time
  control=c
 elif event['kind']=='native':
  record=event['native'];raw=base64.b64decode(record['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=record['prefix_sha256']:raise ValueError('Native bytes changed')
  v=decode_physics_prefix(raw,record['received_monotonic_seconds'])
  if v.packet_id!=record['packet_id'] or v.packet_id<=last or v.received_monotonic_seconds<last_clock or json.loads(json.dumps(dict(v.fields)))!=record['fields']:raise ValueError('Native data/order mismatch')
  last,last_clock=v.packet_id,v.received_monotonic_seconds;status=event['native_ai'];raw_ai=base64.b64decode(status['prefix_base64'],validate=True)
  if hashlib.sha256(raw_ai).hexdigest()!=status['prefix_sha256'] or AI_STATUS_PREFIX.unpack(raw_ai)[1]!=0 or status['builtin_ai_controlled'] is not False:raise ValueError('AI flag mismatch')
  if diagnostic(raw)!=event['diagnostic'] or event['diagnostic']['numberOfTyresOut']!=0 or abs(v.fields['speedKmh'])>8:raise ValueError('Recorded motion envelope invalid')
  samples+=1;maxspeed=max(maxspeed,abs(v.fields['speedKmh']));first=first or v.packet_id;last_sample=v
  if control and control['steering'] and v.received_monotonic_seconds-control_since>=.1:
   key='positive' if control['steering']>0 else 'negative';groups[key].append((v.fields['steerAngle'],v.fields['localAngularVel'][2],v.fields['speedKmh'],v.fields['localAngularVel'][1]))
if not last_sample or abs(last_sample.fields['speedKmh'])>=.1 or last_sample.fields['brake']<=.98 or not all(groups.values()):raise ValueError('Stopped source and both observed pulse samples required')
summary={k:{'samples':len(v),'native_steer_median':statistics.median(x[0] for x in v),
 'native_local_angular_z_median':statistics.median(x[1] for x in v),'native_local_angular_y_median':statistics.median(x[3] for x in v),'speed_kmh_median':statistics.median(x[2] for x in v)} for k,v in groups.items()}
result=dict(status='completed_low_speed_raw_audit',samples=samples,first_packet=first,last_packet=last,max_speed_kmh=maxspeed,
 pulse_groups=summary,post_stop_verified=True,actuator_messages_sent=0,actor_deployed=False,training_admissible=False,
 physical_axis_units_and_time_verified=False,causal_closed_loop_driving_verified=False)
write_json(run/'result.json',result);print(json.dumps(result))
