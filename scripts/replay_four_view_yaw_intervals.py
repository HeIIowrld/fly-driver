"""Offline four-view yaw intervals from original camera bounds and packet receipts."""
import argparse,base64,hashlib,json,zipfile
from bisect import bisect_left
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_angular_history import NativeAngularHistory
from flybrain_drive.view_yaw_interval import prepare_view_yaw_interval
from flybrain_drive.virtual_body import VirtualBodyEncoder
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
VIEWS=('front','left','right','rear')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',choices=('20261101-fast-body-fourview-probe','20261101-low-speed-fourview-prefix-reverse-probe'),required=True);p.add_argument('--output',required=True);a=p.parse_args();root=B/a.source;r=read(root/'result.json');bundle=read(root/'vision/bundle.json');archive=root/'vision/camera.zip';catalog={x['file']:x['sha256'] for x in read(root/'vision/archive.json')}
 if r['status']!='completed_low_speed_pulse_diagnostic' or r['training_admissible'] is not False or r['camera_close']['producer_terminal']['status']!='stopped' or sha256(archive)!=bundle['sha256'] or r['camera_archive']['bundle_sha256']!=bundle['sha256'] or sha256(root/'vision/archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('exact terminal diagnostic source required')
 packets={};frames={}
 with (root/'events.jsonl').open(encoding='utf8') as file:
  for line in file:
   e=json.loads(line)
   if e.get('kind')!='native':continue
   n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
   if h!=n['prefix_sha256']:raise ValueError('raw native hash mismatch')
   sample=decode_physics_prefix(raw,float(n['received_monotonic_seconds']));pid=int(n['packet_id'])
   if sample.packet_id!=pid or (pid in packets and packets[pid]['hash']!=h):raise ValueError('original packet mismatch')
   packets.setdefault(pid,{'time':sample.received_monotonic_seconds,'hash':h,'sample':sample})
   camera=e.get('camera')
   if camera is not None:
    idx=int(camera['capture_id'])
    if idx in frames and frames[idx]['file_sha256']!=camera['file_sha256']:raise ValueError('camera capture changed')
    frames.setdefault(idx,camera)
 packets=sorted(packets.values(),key=lambda x:x['time']);receipts=[x['time'] for x in packets]
 if len(receipts)<2 or np.any(np.diff(receipts)<=0):raise ValueError('strict original first receipts required')
 frame_rows=[]
 with zipfile.ZipFile(archive) as z:
  for idx,camera in sorted(frames.items()):
   filename=Path(camera['file']).name;data=z.read(filename);h=hashlib.sha256(data).hexdigest()
   if h!=camera['file_sha256'] or h!=catalog[filename]:raise ValueError('camera metadata hash mismatch')
   m=json.loads(data)
   if m['generation']!=camera['generation'] or int(m['index'])!=idx:raise ValueError('capture identity mismatch')
   event_views={x['view_id']:x for x in camera['views']};meta_views={x['view_id']:x for x in m['views']}
   if set(event_views)!=set(VIEWS) or set(meta_views)!=set(VIEWS):raise ValueError('exact four views required')
   bounds=np.array([float(event_views[v]['render_started_lower']) for v in VIEWS]);received=float(camera['received'])
   if not np.isfinite(bounds).all() or np.any(bounds>received):raise ValueError('ordered per-view bounds required')
   frame_rows.append({'id':idx,'bounds':bounds,'received':received})
 encoder=VirtualBodyEncoder('configs/ac-virtual-body-v1.json',native_identity=read('configs/ac-virtual-body-v1.json')['native_identity'])
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source=a.source,events_sha256=sha256(root/'events.jsonl'),camera_sha256=bundle['sha256'],views=list(VIEWS),policies=['at_camera','wait_up_to_50ms'],optimizer_updates=0,training_admissible=False))
 result={'status':'failed','axis_validity_changed':False,'optimizer_updates':0,'vehicle_control_sent':False}
 try:
  pairs=[(a,b) for a,b in zip(frame_rows[:-1],frame_rows[1:]) if b['id']==a['id']+1];count=len(pairs)
  if count==0:raise ValueError('consecutive frame groups required')
  means=np.full((2,count,4),np.nan);angles=np.full_like(means,np.nan);valid=np.zeros((2,count,4),dtype=bool);first=np.full_like(means,np.nan);last=np.full_like(means,np.nan);decisions=np.full((2,count),np.nan);reasons=[{},{}];waits=[]
  for policy in range(2):
   history=NativeAngularHistory(capacity=1024);cursor=0;previous_decision=-1.
   for i,(before,after) in enumerate(pairs):
    camera=after['received'];decision=camera
    if policy==1:
     choices=[]
     for end in after['bounds']:
      k=bisect_left(receipts,float(end))
      if k<len(receipts) and receipts[k]<=camera+.05:choices.append(receipts[k])
     if choices:decision=max(camera,max(choices))
    if decision<previous_decision:raise ValueError('decision sequence reversed')
    while cursor<len(packets) and receipts[cursor]<=decision:
     history.offer(packets[cursor]['sample']);cursor+=1
    decisions[policy,i]=decision
    if policy==1:waits.append(decision-camera)
    for j in range(4):
     try:
      y=prepare_view_yaw_interval(history,encoder,previous_render_lower=float(before['bounds'][j]),current_render_lower=float(after['bounds'][j]),camera_receipt=camera,decision=decision)
     except ValueError as error:
      reason=str(error);reasons[policy][reason]=reasons[policy].get(reason,0)+1;continue
     if y.last_receipt>decision or y.actor_qualified or y.physical_time_verified:raise ValueError('unsupported future or qualification')
     means[policy,i,j]=y.yaw_left_mean_rad_s;angles[policy,i,j]=y.yaw_left_integral_rad;first[policy,i,j]=y.first_receipt;last[policy,i,j]=y.last_receipt;valid[policy,i,j]=True
    previous_decision=decision
  common=valid[0]&valid[1]
  if np.any(valid[0]&~valid[1]):raise ValueError('bounded waiting lost a previously valid interval')
  agreement=float(np.max(np.abs(means[0][common]-means[1][common]))) if common.any() else None
  if agreement is not None and agreement>1e-12:raise ValueError('waiting changed an already available interval')
  out=run/'intervals.npz';np.savez_compressed(out,means=means,angles=angles,valid=valid,first_receipt=first,last_receipt=last,decisions=decisions,frame_before=np.array([x[0]['id'] for x in pairs]),frame_after=np.array([x[1]['id'] for x in pairs]),start_bounds=np.stack([x[0]['bounds'] for x in pairs]),end_bounds=np.stack([x[1]['bounds'] for x in pairs]),camera_receipt=np.array([x[1]['received'] for x in pairs]))
  result.update(status='completed_four_view_yaw_interval_offline_replay',pairs=count,views=list(VIEWS),valid_by_policy_and_view=valid.sum(axis=1).tolist(),complete_four_views_by_policy=valid.all(axis=2).sum(axis=1).tolist(),unavailable_reasons=reasons,wait_median_ms=float(np.median(waits)*1000),wait_max_ms=float(np.max(waits)*1000),wait_nonzero_count=int(np.count_nonzero(waits)),common_max_abs=agreement,output_sha256=sha256(out),actor_qualified=False,physical_time_verified=False,interpretation='Diagnostic only; original host render lower bounds, no physical exposure alignment, no circuit or live control')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_four_view_yaw_interval_offline_replay':raise SystemExit(1)
if __name__=='__main__':main()
