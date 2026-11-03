"""Apply causal native angular averaging to previously reviewed archived sessions."""
import argparse,base64,hashlib,json,zipfile
from pathlib import Path
from bisect import bisect_left
import numpy as np
from scipy.spatial.transform import Rotation
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.causal_native_angular_average import integrate_received_angular_velocity
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();review=Path('runs/ac-virtual-body-geometry-20261029-t1026/result.json');prior=read(review)
 if prior['status']!='completed' or len(prior['recordings'])!=2:raise ValueError('two exact archived geometry sources required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),original_review_sha256=sha256(review),window_frames=5,native_max_gap_s=.05,method='front only, time weighted first receipt native values, decision after final bracketing receipt',optimizer_updates=0));reports=[];arrays={}
 try:
  for ri,entry in enumerate(prior['recordings']):
   root=Path(entry['recording']);ef=root/'events.jsonl';bundle=read(root/'vision/bundle.json');archive=root/'vision'/bundle['file']
   if sha256(ef)!=entry['events_sha256'] or sha256(archive)!=entry['archive_sha256'] or bundle['sha256']!=entry['archive_sha256']:raise ValueError('original archive hash mismatch')
   packets={};frames={}
   with ef.open(encoding='utf8') as file:
    for line in file:
     e=json.loads(line);kind=e.get('kind');engine=kind is None and all(k in e for k in ('native','camera','decision_monotonic_seconds'))
     if kind in ('raw_pair','expert_raw_native') or engine:
      n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
      if h!=n['prefix_sha256']:raise ValueError('native hash mismatch')
      t=float(n['received_monotonic_seconds']);sample=decode_physics_prefix(raw,t);pid=int(n['packet_id'])
      if pid!=sample.packet_id:raise ValueError('native packet mismatch')
      if pid in packets and packets[pid]['hash']!=h:raise ValueError('repeated native changed')
      packets.setdefault(pid,{'time':t,'hash':h,'gyro':sample.fields['localAngularVel']})
     camera=e['actor_input']['image'] if kind=='student_request' else e['camera'] if kind=='expert_observation' or engine else None
     if camera is not None:
      idx=int(camera['capture_id'])
      if idx in frames and frames[idx]['file_sha256']!=camera['file_sha256']:raise ValueError('camera metadata changed')
      frames.setdefault(idx,camera)
   packets=sorted(packets.values(),key=lambda v:v['time']);receipt=[x['time'] for x in packets]
   if len(receipt)<2 or np.any(np.diff(receipt)<=0):raise ValueError('distinct ordered first receipt required')
   gyros=np.asarray([x['gyro'] for x in packets])[:,[2,0,1]];items=[]
   with zipfile.ZipFile(archive) as z:
    for index,camera in sorted(frames.items()):
     raw=z.read(f'frame-{index:04d}.json')
     if hashlib.sha256(raw).hexdigest()!=camera['file_sha256']:raise ValueError('camera metadata hash mismatch')
     meta=json.loads(raw);front=next(v for v in meta['views'] if v['view_id']=='front');front_event=next(v for v in camera['views'] if v['view_id']=='front');look=np.asarray(front['renderer_only_world_look']);up=np.asarray(front['renderer_only_world_up']);matrix=np.column_stack([look,np.cross(up,look),up]);lower=float(front_event['render_started_lower'])
     if not np.allclose(matrix.T@matrix,np.eye(3),atol=2e-5) or not np.isclose(np.linalg.det(matrix),1.,atol=2e-5):raise ValueError('orthonormal front basis required')
     items.append({'index':index,'clock':float(front['render_started_clock']),'lower':lower,'matrix':matrix})
   x=[];y=[];delays=[]
   for i in range(2,len(items)-2):
    w=items[i-2:i+3];ts=np.array([t['clock'] for t in w]);dt=ts[-1]-ts[0];lo,hi=w[0]['lower'],w[-1]['lower']
    if not .02<dt<.2 or not np.all(np.diff(ts)>0) or not lo<hi or abs((hi-lo)-dt)>.01:continue
    if any(w[j+1]['index']!=w[j]['index']+1 for j in range(4)):continue
    k=bisect_left(receipt,hi)
    if k>=len(receipt) or receipt[k]-hi>.05:continue
    try:avg=integrate_received_angular_velocity(receipt,gyros,start=lo,end=hi,decision=receipt[k],max_gap=.05)
    except ValueError:continue
    omega=Rotation.from_matrix(w[-1]['matrix']@w[0]['matrix'].T).as_rotvec()/dt;y.append(w[2]['matrix'].T@omega);x.append(avg);delays.append(receipt[k]-hi)
   x=np.asarray(x);y=np.asarray(y);metrics={}
   if len(x)>=5:
    for i,name in enumerate(('roll','pitch','yaw')):
     v=x[:,i];target=y[:,i];slope,offset=np.linalg.lstsq(np.column_stack([v,np.ones(len(v))]),target,rcond=None)[0];corr=float(np.corrcoef(v,target)[0,1]) if np.std(v)>0 and np.std(target)>0 else None;rmse=float(np.sqrt(np.mean((v-target)**2)));span=float(np.ptp(v));metrics[name]={'samples':len(v),'native_range':span,'slope':float(slope),'correlation':corr,'rmse':rmse,'existing_gate_passed':bool(corr is not None and span>=.02 and .9<=slope<=1.1 and corr>=.9 and rmse<.005)}
   reports.append({'recording':str(root),'input_packets':len(packets),'camera_groups':len(items),'paired_windows':len(x),'metrics':metrics,'availability_delay_median_ms':float(np.median(delays)*1000) if delays else None,'events_sha256':entry['events_sha256'],'archive_sha256':entry['archive_sha256']});arrays[f'native_{ri}']=x;arrays[f'renderer_{ri}']=y;write_json(run/'progress.json',{'completed_recording':str(root),'windows':len(x)})
  out=run/'comparisons.npz';np.savez_compressed(out,**arrays);result={'status':'completed_historical_integrated_rotation_diagnostic','reports':reports,'output_sha256':sha256(out),'source_clock_verified':False,'physical_exposure_sync_verified':False,'axis_validity_changed':False,'training_admissible':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'Same simulator sensors, prior seen recordings, adaptive post-hoc method, overlapping windows and unverified physical time; no admission of axes.'}
 except Exception as e:result={'status':'failed','error':type(e).__name__,'reason':str(e),'reports':reports,'axis_validity_changed':False,'optimizer_updates':0,'vehicle_control_sent':False}
 write_json(run/'result.json',result);print(json.dumps(result));
 if result['status']!='completed_historical_integrated_rotation_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
