"""Compare lateral sign proxies in the same preserved corner recordings."""
import base64,hashlib,json,struct,zipfile
from bisect import bisect_left,bisect_right
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
BASE=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');PLAN=Path('D:/CodexResearchRuns/flybrain-corner-axis-corrected-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-corner-axis-corrected-20261102-02')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def unit(v):
 a=np.asarray(v,float);n=np.linalg.norm(a)
 if n<1e-8 or not np.isfinite(a).all():raise ValueError('invalid basis')
 return a/n
def read(role,expected,event_hash):
 root=BASE/f'20261031-pace-{role}-native-copy-candidate-00/recording';r=json.loads((root/'result.json').read_text());native=root/'native-copy-history.json';events=root/'events.jsonl';bundle=json.loads((root/'vision/bundle.json').read_text());archive=root/'vision/camera.zip';catalog=root/'vision/archive.json'
 if sha(root/'result.json')!=expected or r['status']!='completed' or r['training_admissible'] is not False or sha(native)!=r['native_copy_history']['sha256'] or sha(events)!=event_hash or sha(archive)!=bundle['sha256'] or bundle['sha256']!=r['vision_archive']['bundle_sha256'] or sha(catalog)!=bundle['archive_manifest_sha256']:raise ValueError('source provenance')
 hashes={x['file']:x['sha256'] for x in json.loads(catalog.read_text())};seen={};data=[]
 for row in json.loads(native.read_text()):
  a=row['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True);digest=hashlib.sha256(raw).hexdigest()
  if digest!=a['prefix_sha256']:raise ValueError('native digest')
  sample=decode_teacher_physics(raw,a['received_monotonic_seconds']);pid=sample.physics.packet_id
  if pid!=a['packet_id']:raise ValueError('native id')
  if pid in seen:
   if seen[pid]!=digest:raise ValueError('changed same packet')
   continue
  seen[pid]=digest;v=sample.physics.fields;data.append([sample.physics.received_monotonic_seconds,v['speedKmh']/3.6,v['localAngularVel'][1],*struct.unpack_from('<3f',raw,32),*v['accG']])
 data=np.asarray(data);times=data[:,0];tlist=times.tolist()
 if np.any(np.diff(times)<=0):raise ValueError('native order')
 cameras={}
 with events.open(encoding='utf8') as f:
  for line in f:
   event=json.loads(line);camera=event.get('camera')
   if not camera:continue
   idx=camera['capture_id']
   if idx in cameras:
    if cameras[idx]['file_sha256']!=camera['file_sha256']:raise ValueError('frame changed')
    continue
   cameras[idx]=camera
 rows=[]
 with zipfile.ZipFile(archive) as z:
  for idx,camera in sorted(cameras.items()):
   name=Path(camera['file']).name;raw=z.read(name);digest=hashlib.sha256(raw).hexdigest()
   if digest!=camera['file_sha256'] or hashes.get(name)!=digest:raise ValueError('camera metadata digest')
   frame=json.loads(raw);view=next(x for x in frame['views'] if x['view_id']=='front');lower=next(x['render_started_lower'] for x in camera['views'] if x['view_id']=='front');i=bisect_left(tlist,lower)-1
   if i<0 or lower-times[i]>.05 or data[i,1]<5 or abs(data[i,2])<.02:continue
   t=times[i];lo=bisect_left(tlist,t-.075);hi=bisect_right(tlist,t+.075);window=times[lo:hi]
   if len(window)<8 or window[0]>t-.06 or window[-1]<t+.06 or np.max(np.diff(window))>.02:continue
   d=window-t;d=d-d.mean();vel=data[lo:hi,3:6];world=(d[:,None]*(vel-vel.mean(axis=0))).sum(axis=0)/(d@d)
   forward=unit(view['renderer_only_world_look']);up=unit(view['renderer_only_world_up']);left=unit(np.cross(up,forward));up=unit(np.cross(forward,left))
   if float(left@unit(next(v['renderer_only_world_look'] for v in frame['views'] if v['view_id']=='left')))<.99:raise ValueError('computed left differs from recorded left view')
   rows.append({'capture_id':idx,'speed_m_s':float(data[i,1]),'yaw_rad_s':float(data[i,2]),'body_receipt_age_ms':float((lower-t)*1000),'native_axis0_m_s2':float(data[i,6]*9.80665),'velocity_yaw_proxy_m_s2':float(data[i,1]*data[i,2]),'projected_left_m_s2':float(world@left),'projected_forward_m_s2':float(world@forward),'projected_up_m_s2':float(world@up)})
 if len(rows)<20:raise ValueError('insufficient matched turning frames')
 a=np.array([x['native_axis0_m_s2'] for x in rows]);b=np.array([x['velocity_yaw_proxy_m_s2'] for x in rows]);c=np.array([x['projected_left_m_s2'] for x in rows]);
 return {'frames':len(rows),'positive_yaw_frames':sum(x['yaw_rad_s']>0 for x in rows),'negative_yaw_frames':sum(x['yaw_rad_s']<0 for x in rows),'native_vs_projected_left_correlation':float(np.corrcoef(a,c)[0,1]),'native_vs_speed_yaw_correlation':float(np.corrcoef(a,b)[0,1]),'speed_yaw_vs_projected_left_correlation':float(np.corrcoef(b,c)[0,1]),'native_axis_std_m_s2':float(a.std()),'speed_yaw_std_m_s2':float(b.std()),'projected_left_std_m_s2':float(c.std()),'maximum_native_receipt_age_ms':max(x['body_receipt_age_ms'] for x in rows),'rows':rows}
def main():
 p=json.loads(PLAN.read_text());runs={role:read(role,expected,event) for role,expected,event in zip(('train','development'),p['result_sha256'],p['event_sha256'],strict=True)};report={'status':'completed_exploratory_same_corner_axis_crosscheck','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'runs':runs,'physical_time_verified':False,'acceleration_axes_registered':False,'training_admissible':False,'actor_qualified':False,'limitations':p['limitations']};OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in runs.items()},indent=2))
if __name__=='__main__':main()
