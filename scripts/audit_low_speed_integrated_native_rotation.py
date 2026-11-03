"""Exploratory native angular-rate versus renderer rotation on archived low-speed probe."""
import argparse,base64,hashlib,json,zipfile
from pathlib import Path
from bisect import bisect_left
import numpy as np
from scipy.spatial.transform import Rotation
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.ac_native_physics import decode_physics_prefix
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=B/'20261101-low-speed-fourview-stationary-prefix-probe';r=read(root/'result.json');bundle=read(root/'vision/bundle.json');archive=root/'vision/camera.zip';catalog=read(root/'vision/archive.json'); hashes={x['file']:x['sha256'] for x in catalog}
 if r['status']!='completed_low_speed_pulse_diagnostic' or r['training_admissible'] is not False or r['camera_close']['producer_terminal']['status']!='stopped' or sha256(archive)!=bundle['sha256'] or bundle['sha256']!=r['camera_archive']['bundle_sha256'] or sha256(root/'vision/archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('exact terminal diagnostic source required')
 packets={};frames={}
 with (root/'events.jsonl').open(encoding='utf8') as f:
  for line in f:
   e=json.loads(line)
   if e.get('kind')!='native':continue
   n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
   if h!=n['prefix_sha256']:raise ValueError('raw packet hash mismatch')
   t=float(n['received_monotonic_seconds']);sample=decode_physics_prefix(raw,t);pid=int(n['packet_id'])
   if sample.packet_id!=pid:raise ValueError('packet identity mismatch')
   if pid in packets and packets[pid]['hash']!=h:raise ValueError('duplicate packet changed')
   packets.setdefault(pid,{'time':t,'hash':h,'gyro':sample.fields['localAngularVel']})
   camera=e.get('camera')
   if camera is not None:
    index=int(camera['capture_id'])
    if index in frames and frames[index]['file_sha256']!=camera['file_sha256']:raise ValueError('capture changed')
    frames.setdefault(index,camera)
 packets=sorted(packets.values(),key=lambda x:x['time']);receipt=[x['time'] for x in packets];items=[]
 with zipfile.ZipFile(archive) as z:
  for index,camera in sorted(frames.items()):
   filename=Path(camera['file']).name;raw=z.read(filename);h=hashlib.sha256(raw).hexdigest()
   if hashes.get(filename)!=h or camera['file_sha256']!=h:raise ValueError('camera metadata hash mismatch')
   meta=json.loads(raw);views={v['view_id']:v for v in meta['views']};front=views['front'];look=np.asarray(front['renderer_only_world_look'],dtype=float);up=np.asarray(front['renderer_only_world_up'],dtype=float);matrix=np.column_stack([look,np.cross(up,look),up]);lower=next(v['render_started_lower'] for v in camera['views'] if v['view_id']=='front');pos=bisect_left(receipt,lower)-1
   if pos<0 or lower-receipt[pos]>.05:continue
   if not np.allclose(matrix.T@matrix,np.eye(3),atol=2e-5) or not np.isclose(np.linalg.det(matrix),1.,atol=2e-5):raise ValueError('orthonormal renderer basis required')
   items.append({'index':index,'clock':float(front['render_started_clock']),'host_lower':lower,'matrix':matrix,'gyro':np.asarray(packets[pos]['gyro'])[[2,0,1]],'age':lower-receipt[pos]})
 native=[];render=[]
 for i in range(1,len(items)-1):
  window=items[i-1:i+2];ts=np.array([x['clock'] for x in window]);dt=ts[-1]-ts[0]
  if not .02<dt<.2 or not np.all(np.diff(ts)>0) or any(window[j+1]['index']!=window[j]['index']+1 for j in range(2)):continue
  lo,hi=window[0]['host_lower'],window[-1]['host_lower']
  if not lo<hi or abs((hi-lo)-dt)>.01:continue
  k0=bisect_left(receipt,lo);k1=bisect_left(receipt,hi)
  if k0<1 or k1>=len(receipt) or lo-receipt[k0-1]>.05 or receipt[k1]-hi>.05:continue
  sample_t=np.array([lo]+receipt[k0:k1]+[hi]); gyro_all=np.asarray([p['gyro'] for p in packets])[:,[2,0,1]]
  sample_v=np.column_stack([np.interp(sample_t,receipt,gyro_all[:,j]) for j in range(3)])
  if np.any(np.diff(sample_t)<=0):continue
  integrated=np.sum((sample_v[:-1]+sample_v[1:])*np.diff(sample_t)[:,None]/2,axis=0)/(hi-lo)
  world=Rotation.from_matrix(window[-1]['matrix']@window[0]['matrix'].T).as_rotvec()/dt;render.append(window[1]['matrix'].T@world);native.append(integrated)
 native=np.asarray(native);render=np.asarray(render)
 if native.ndim!=2 or native.shape[1]!=3 or native.shape!=render.shape or len(native)<5:
  run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(root/'events.jsonl'),camera_bundle_sha256=bundle['sha256'],strict_before_render_native_max_age=.05,centered_window_frames=3,basis='front look cross front up',native_rate='trapezoidal all first-receipt packets in renderer bracket; diagnostic future packet permitted',source_training_admissible=False,optimizer_updates=0))
  report={'status':'insufficient_contiguous_timing_windows','eligible_groups':len(items),'paired_windows':len(native),'indices':[x['index'] for x in items],'script_clocks':[x['clock'] for x in items],'source_clock_verified':False,'physical_exposure_sync_verified':False,'axis_validity_changed':False,'training_admissible':False,'vehicle_control_sent':False}
  write_json(run/'result.json',report);print(json.dumps(report));raise SystemExit(1)
 metrics={}
 for i,name in enumerate(('roll','pitch','yaw')):
  x=native[:,i];y=render[:,i];slope,offset=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0];corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None;rmse=float(np.sqrt(np.mean((x-y)**2)));span=float(np.ptp(x));passed=bool(corr is not None and span>=.02 and .9<=slope<=1.1 and corr>=.9 and rmse<.005);metrics[name]={'samples':len(x),'native_range':span,'slope':float(slope),'correlation':corr,'rmse':rmse,'existing_gate_passed':passed}
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(root/'events.jsonl'),camera_bundle_sha256=bundle['sha256'],strict_before_render_native_max_age=.05,centered_window_frames=3,basis='front look cross front up',native_rate='trapezoidal all first-receipt packets in renderer bracket; diagnostic future packet permitted',source_training_admissible=False,optimizer_updates=0));np.savez_compressed(run/'angular-comparison.npz',native=native,renderer=render)
 result={'status':'completed_integrated_native_single_view_comparison','metrics':metrics,'eligible_groups':len(items),'paired_windows':len(native),'comparison_sha256':sha256(run/'angular-comparison.npz'),'source_clock_verified':False,'physical_exposure_sync_verified':False,'axis_validity_changed':False,'training_admissible':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'Retrospective interpolation may use later packets; host render bounds are not physical timestamps. Same simulator and overlapping windows; masks unchanged.'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
