"""Match observed CSP cached numeric states to native packet values; diagnostic only."""
import base64,hashlib,json,struct
from collections import defaultdict
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.diagnostic_cached_body_clock import summarize_cached_body_around_render
ROOT=Path('D:/CodexResearchRuns/flybrain-marker-motion-20261102-02');OUT=Path('D:/CodexResearchRuns/flybrain-cached-native-value-match-20261102-02')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def key(x):return tuple(struct.pack('<f',float(v)) for v in x)
def main():
 result=json.loads((ROOT/'result.json').read_text(encoding='utf8'));clock=json.loads((ROOT/'clock-after.json').read_text(encoding='utf8'));records=json.loads((ROOT/'native-copy-windows.json').read_text(encoding='utf8'));frames=json.loads((ROOT/'diagnostic-frames.json').read_text(encoding='utf8'));observed=[json.loads(x) for x in (ROOT/'live-frame-hashes.jsonl').read_text(encoding='utf8').splitlines()]
 for f,k in [('native-copy-windows.json','native_copy_sha256'),('clock-after.json','clock_after_sha256'),('diagnostic-frames.json','metadata_sha256'),('live-frame-hashes.jsonl','host_observed_hashes_sha256')]:
  if sha(ROOT/f)!=result[k]:raise ValueError('source hash mismatch')
 if result['status']!='completed' or len(frames)!=len(observed) or len(frames)!=result['frames']:raise ValueError('incomplete source')
 # Distinct IDs within an exact binary32 value tuple; multiple copies are retained.
 index=defaultdict(list)
 for rec in records:
  a=rec['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('native hash mismatch')
  sample=decode_teacher_physics(raw,rec['copy']['second_copy_end']);v=sample.physics.fields
  if sample.physics.packet_id!=a['packet_id']:raise ValueError('packet id mismatch')
  index[key([v['speedKmh'],*v['localAngularVel']])].append((sample.physics.packet_id,sample.physics.received_monotonic_seconds,key(v['accG'])))
 rows=[];offset=(clock['offset_lower_seconds']+clock['offset_upper_seconds'])/2
 for item,obs in zip(frames,observed,strict=True):
  path=Path(item['source_file']);i=item['index']
  if i!=obs['index'] or sha(path)!=item['sha256'] or item['sha256']!=obs['metadata_sha256']:raise ValueError('frame hash mismatch')
  frame=json.loads(path.read_text(encoding='utf8'));summarize_cached_body_around_render(frame)
  if frame['request_id']!=result['request_id'] or frame['generation']!=clock['generation'] or frame['aid_probe_only'] is not True:raise ValueError('diagnostic identity')
  for view in frame['views']:
   e=view['render_marker_diagnostic'];name=view['view_id'];image=obs['images'][name]
   if (e['index']!=i or e['view_id']!=name or not view['render_started_clock']<=e['source_clock']<=view['render_call_finished_clock'] or sha(path.parent/image['file'])!=image['sha256']):raise ValueError('marker/image mismatch')
   if e['speed_kmh']<=1:continue
   hits=index.get(key([e['speed_kmh'],*e['local_angular_velocity']]),[]);ids=sorted({v[0] for v in hits});full=[v for v in hits if v[2]==key(e['acceleration_g'])];estimated=e['source_clock']+offset
   rows.append({'frame':i,'view':name,'speed_kmh':e['speed_kmh'],'matching_packet_count':len(ids),'packet_ids':ids,'matching_all_three_fields':bool(full),'closest_matching_copy_minus_callback_s':min((t-estimated for _,t,_ in hits),key=abs) if hits else None})
 if not rows:raise ValueError('no moving values')
 valid=[x for x in rows if x['matching_packet_count']];unique=[x for x in rows if x['matching_packet_count']==1];dt=np.array([x['closest_matching_copy_minus_callback_s'] for x in valid]);OUT.mkdir(parents=True,exist_ok=False)
 report={'status':'completed_exploratory_cached_native_value_match','moving_view_observations':len(rows),'speed_and_gyro_exact_binary32_matches':len(valid),'unique_packet_id_for_speed_and_gyro':len(unique),'also_acceleration_exact_binary32_matches':sum(x['matching_all_three_fields'] for x in rows),'matching_copy_minus_estimated_callback_ms':{'min':float(dt.min()*1000),'median':float(np.median(dt)*1000),'max':float(dt.max()*1000)},'clock_interval_width_ms':1000*(clock['offset_upper_seconds']-clock['offset_lower_seconds']),'source_result_sha256':sha(ROOT/'result.json'),'program_sha256':sha(Path(__file__)),'rows':rows,'physical_exposure_sync_verified':False,'native_packet_identity_verified':False,'training_admissible':False,'actor_qualified':False,'limits':'Posthoc exact binary32 value comparison on one short diagnostic. Same numeric values may share a cache or recur; packet copied at host is not source generation time; marker callback is not proof of world render physics tick. Observations temporally and across views correlated; no causal or independent timing validation.'}
 (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
