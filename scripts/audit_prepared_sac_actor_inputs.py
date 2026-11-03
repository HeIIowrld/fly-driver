"""Independently join prepared actor arrays back to immutable qualified source bytes.

No neural weights, controls, simulator, lifecycle or renderer-truth actor inputs.
"""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
from scripts.audit_synchronized_sac_demonstration import native_samples

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 data_root=Path(a.data);meta=read(data_root/'manifest.json')['config'];dr=read(data_root/'result.json');contract=read(data_root/'contract.json')
 root=Path(meta['arguments']['recording']);q=Path(meta['arguments']['qualification']);qr=read(q/'result.json')
 run=begin_run(a.output,dict(arguments=vars(a),seed=780208,program_sha256=sha256(__file__),data_result_sha256=sha256(data_root/'result.json'),data_manifest_sha256=sha256(data_root/'manifest.json')))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False)
 try:
  if (dr['status']!='completed_qualified_actor_preprocessing' or not qr['training_admissible']
      or sha256(data_root/'actor-only.npz')!=dr['arrays_sha256'] or sha256(data_root/'contract.json')!=dr['contract_sha256']
      or sha256(q/'result.json')!=contract['source_qualification_sha256'] or contract['unqualified_pixel_inputs']!=0):raise ValueError('Independent qualification or frozen prepared hashes differ')
  body,body_times=native_samples(root/'camera/native-body.jsonl',qr['native_body_sha256'])
  identity=read(meta['arguments']['identity'])['identity'];encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity)
  encoded=[]
  for b in body:encoded.append(encoder.encode(decode_navigation_body_prefix(b['prefix'],b['receipt']),episode_id=contract['episode_id'],decision_monotonic_seconds=b['receipt']))
  packet_index={b['packet_id']:j for j,b in enumerate(body)}
  with CameraArchive(root/'camera/vision') as archive:frames=[json.loads(s) for s in archive.read('stream.jsonl').decode('utf8').splitlines()]
  events=[json.loads(s) for s in (root/'teacher/events.jsonl').read_text(encoding='utf8').splitlines()];requests=[e for e in events if e['kind']=='teacher_request'];pairings=read(q/'pairings.json')
  lap=read(q/'lap-proof/result.json');begin,end=lap['crossings'][0]['host_time'],lap['crossings'][-1]['host_time']
  with np.load(data_root/'actor-only.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
  expected={'scene','body','body_valid','view_body','view_body_valid','decision_time','body_receipt','view_receipts','render_lowers','camera_receipt','labels','camera_index','body_packet','view_packets','fit_mask','label_valid','teacher_request_index'}
  if set(d)!=expected or not np.isfinite(d['scene']).all():raise ValueError('Only declared finite actor arrays and audit indices allowed')
  seen_scenes={};label_indices=[];native_events=0
  for n,t in enumerate(d['decision_time']):
   i=int(d['camera_index'][n]);j=packet_index[int(d['body_packet'][n])];obs=encoded[j];frame=frames[i]
   if (obs.original_received_monotonic_seconds!=d['body_receipt'][n] or not 0<=t-d['body_receipt'][n]<=.05
       or d['camera_receipt'][n]!=frame['received'] or not frame['received']<=t
       or np.searchsorted(body_times,t,side='right')-1!=j):raise ValueError('Original body/camera receipt or latest causal selection changed')
   if not np.array_equal(d['body'][n],obs.values) or not np.array_equal(d['body_valid'][n],obs.valid):raise ValueError('Actor body21 or explicit mask differs from original encoder')
   views={v['view_id']:v for v in frame['views']}
   for v,name in enumerate(('front','left','right','rear')):
    h=encoded[packet_index[int(d['view_packets'][n,v])]];lower=views[name]['render_started_lower']
    if (d['render_lowers'][n,v]!=lower or not 0<=t-lower<=.15 or not h.original_received_monotonic_seconds<lower
        or h.original_received_monotonic_seconds!=d['view_receipts'][n,v]
        or not np.array_equal(h.values,d['view_body'][n,v]) or not np.array_equal(h.valid,d['view_body_valid'][n,v])):raise ValueError('Per-view original historical body or render bound changed')
   if i in seen_scenes and not np.array_equal(d['scene'][n],seen_scenes[i]):raise ValueError('Repeated frozen visual frame has decision-dependent features')
   seen_scenes[i]=d['scene'][n]
   k=int(d['teacher_request_index'][n]);valid=bool(d['label_valid'][n]);fit=bool(d['fit_mask'][n])
   if valid:
    request=requests[k];control=request['request']['requested_control'];target=np.array([control['steering']/.6,control['throttle'],control['brake']])
    if t!=request['decision_started_host_time'] or not np.array_equal(d['labels'][n],target):raise ValueError('Original teacher label/time changed')
    if i!=pairings[k]['camera_index'] or j!=pairings[k]['body_index'] or not pairings[k]['eligible']:raise ValueError('Independent qualified label join differs')
    if fit!=(begin<=request['host_time']<=end) or fit and request['unlimited']!=request['limited']:raise ValueError('Restricted preparation controls entered fit targets')
    label_indices.append(k)
   else:
    if k!=-1 or fit or np.any(d['labels'][n]!=0) or t!=body[j]['receipt']:raise ValueError('Native-only observation invented or retimed a teacher target')
    native_events+=1
  if label_indices!=list(range(contract['source_start_request'],contract['source_end_request'])):raise ValueError('Teacher timeline was spliced or labels omitted')
  held=d['decision_time'][1:]-d['body_receipt'][:-1]
  if np.any(held>.05):raise ValueError('Original held body age exceeds existing50ms guard')
  result.update(status='completed_causal_actor_input_audit',arrays_sha256=dr['arrays_sha256'],contract_sha256=dr['contract_sha256'],
   model_events=len(d['scene']),native_only_events=native_events,original_teacher_labels=len(label_indices),supervised_full_lap_labels=int(d['fit_mask'].sum()),
   maximum_held_body_age_ms=float(held.max()*1000),unqualified_pixel_inputs=0,
   privileged_actor_array_fields=0,original_clocks_and_labels_preserved=True,source_body_encoder_values_exact=True,
   interpretation='Independent raw-byte/clock/label join; frozen FlyVis numerical feature reproduction remains covered by its source/weight hashes, not rerun here')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
