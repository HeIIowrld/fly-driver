"""Audit native angular magnitude for historical stationary windows without assigning axes."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.ac_native_physics import decode_physics_prefix
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=B/'20261029-t1434-body21-flight-cache';m=read(root/'dataset.json');mr=read(root/'result.json')
 if sha256(root/'dataset.json')!=mr['dataset_manifest_sha256']:raise ValueError('original manifest required')
 reports=[]
 for ep in m['episodes']:
  if ep['file'] not in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):continue
  rec=Path(ep['original']['recording']);receipt=read(rec/'result.json');ef=rec/'events.jsonl'
  if receipt.get('training_admissible') is not False or sha256(ef)!=ep['source_events_sha256']:raise ValueError('fit-forbidden source bytes required')
  rows=[];last_packet=None;last_hash=None;first_time=None;last_capture=None
  with ef.open(encoding='utf8') as stream:
   for line in stream:
    e=json.loads(line);n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
    if h!=n['prefix_sha256']:raise ValueError('raw hash mismatch')
    t=float(e['decision_monotonic_seconds']);current=float(n['received_monotonic_seconds']);sample=decode_physics_prefix(raw,current);packet=int(n['packet_id'])
    if packet!=sample.packet_id:raise ValueError('packet mismatch')
    if last_packet is not None and packet<last_packet:raise ValueError('packet reset')
    if packet==last_packet:
     if h!=last_hash:raise ValueError('duplicate packet changed')
    else:first_time=current;last_packet=packet;last_hash=h
    age=t-first_time;camera=e['camera'];capture=camera['capture_id'];fresh_camera=float(camera['received'])<=t;fields=sample.fields;angular=np.asarray(fields['localAngularVel']);wheel=np.asarray(fields['wheelAngularSpeed']);speed=float(fields['speedKmh']);finite=np.isfinite(angular).all() and np.isfinite(wheel).all() and np.isfinite(speed)
    stationary=finite and 0<=age<=.05 and 0<=speed<=.1 and np.max(np.abs(wheel))<=.05 and np.linalg.norm(angular)<=.02
    distinct=capture!=last_capture and (len(rows)==0 or packet!=rows[-1]['packet']);rows.append({'index':e['index'],'decision':t,'packet':packet,'capture_id':capture,'stationary':bool(stationary and fresh_camera),'distinct_from_previous':bool(distinct),'body_age':age,'camera_received':float(camera['received']),'earliest_render_lower':min(float(v['render_started_lower']) for v in camera['views']),'body_precedes_render':first_time<min(float(v['render_started_lower']) for v in camera['views'])});last_capture=capture
  windows=[];group=[]
  for row in rows:
   if not row['stationary'] or (group and (row['decision']-group[-1]['decision']>.15 or not row['distinct_from_previous'])):group=[]
   if row['stationary'] and (not group or row['distinct_from_previous']):group.append(row)
   if len(group)>=11 and group[-1]['decision']-group[0]['decision']>=.4-1e-9:
    windows.append({'first_index':group[0]['index'],'ready_index':row['index'],'duration_s':row['decision']-group[0]['decision'],'count':len(group),'all_body_before_render':all(x['body_precedes_render'] for x in group)});break
  reports.append({'episode':ep['file'],'decisions':len(rows),'host_stationary_decisions':sum(x['stationary'] for x in rows),'qualifying_window':windows[0] if windows else None,'original_events_sha256':ep['source_events_sha256']})
 if len(reports)!=4:raise ValueError('four source episodes required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),dataset_sha256=sha256(root/'dataset.json'),max_body_age=.05,max_speed_kmh=.1,max_wheel_rate=.05,max_angular_vector_norm=.02,min_frames=11,min_span_s=.4,optimizer_updates=0));result={'status':'completed_native_stationary_candidate_audit','reports':reports,'training_admissible':False,'vehicle_control_sent':False,'caveat':'raw vector norm requires no axis-sign registration; host stationary candidates are not physically exposure synchronized and do not make masked roll/pitch valid'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
