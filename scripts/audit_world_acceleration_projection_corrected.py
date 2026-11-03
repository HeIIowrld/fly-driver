"""Diagnostic comparison of native acceleration with world velocity derivative."""
import base64,hashlib,json,struct
from bisect import bisect_left,bisect_right
from collections import defaultdict
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
PLAN=Path('D:/CodexResearchRuns/flybrain-world-accel-corrected-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-world-accel-corrected-20261102-02')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(values):return tuple(struct.pack('<f',float(x)) for x in values)
def unit(x):
 x=np.asarray(x,float);n=np.linalg.norm(x)
 if n<1e-8 or not np.isfinite(x).all():raise ValueError('invalid camera basis')
 return x/n
def analyze(suffix,expected):
 root=Path('D:/CodexResearchRuns')/f'flybrain-marker-motion-20261102-{suffix}';r=json.loads((root/'result.json').read_text(encoding='utf8'))
 if sha(root/'result.json')!=expected or r['status']!='completed':raise ValueError('source identity')
 for f,k in [('native-copy-windows.json','native_copy_sha256'),('diagnostic-frames.json','metadata_sha256')]:
  if sha(root/f)!=r[k]:raise ValueError('source digest')
 rawrows=json.loads((root/'native-copy-windows.json').read_text());frames=json.loads((root/'diagnostic-frames.json').read_text());seen={};records=[];lookup=defaultdict(set)
 for row in rawrows:
  a=row['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('packet hash')
  s=decode_teacher_physics(raw,row['copy']['second_copy_end']);pid=s.physics.packet_id
  if pid!=a['packet_id']:raise ValueError('packet mismatch')
  if pid in seen:continue
  f=s.physics.fields;seen[pid]=len(records);records.append([s.physics.received_monotonic_seconds,*struct.unpack_from('<3f',raw,32),*f['accG']]);lookup[key([f['speedKmh'],*f['localAngularVel']])].add(pid)
 records=np.asarray(records);times=records[:,0];tt=times.tolist()
 if np.any(np.diff(times)<=0):raise ValueError('nonmonotonic timestamps')
 observed=[]
 for fr in frames:
  path=Path(fr['source_file'])
  if sha(path)!=fr['sha256']:raise ValueError('metadata changed')
  meta=json.loads(path.read_text());view=next(v for v in meta['views'] if v['view_id']=='front');e=view['render_marker_diagnostic']
  if meta['training_admissible'] is not False or meta['aid_probe_only'] is not True or e['index']!=fr['index']:raise ValueError('diagnostic marker identity')
  if e['speed_kmh']<=5:continue
  ids=lookup[key([e['speed_kmh'],*e['local_angular_velocity']])]
  if len(ids)!=1:continue
  i=seen[next(iter(ids))];t=times[i];lo=bisect_left(tt,t-.075);hi=bisect_right(tt,t+.075);window=times[lo:hi]
  if len(window)<8 or window[0]>t-.06 or window[-1]<t+.06 or np.max(np.diff(window))>.02:continue
  d=window-t;centered=d-d.mean();world=(centered[:,None]*(records[lo:hi,1:4]-records[lo:hi,1:4].mean(axis=0))).sum(axis=0)/(centered@centered)
  forward=unit(view['renderer_only_world_look']);up=unit(view['renderer_only_world_up']);left=unit(np.cross(up,forward));up=unit(np.cross(forward,left));projected=[float(world@v) for v in (forward,left,up)]
  if float(left@unit(next(v['renderer_only_world_look'] for v in meta['views'] if v['view_id']=='left')))<.99:raise ValueError('computed left differs from recorded left view')
  observed.append({'frame':fr['index'],'packet_id':next(iter(ids)),'native_acceleration_g':records[i,4:7].tolist(),'projected_world_velocity_derivative_m_s2':projected})
 if len(observed)<10:raise ValueError('insufficient matched moving frames')
 a=np.array([x['native_acceleration_g'] for x in observed])*9.80665;b=np.array([x['projected_world_velocity_derivative_m_s2'] for x in observed]);corr=[]
 for i in range(3):corr.append([float(np.corrcoef(a[:,i],b[:,j])[0,1]) for j in range(3)])
 return {'matched_front_frames':len(observed),'correlations_native_rows_forward_left_up_columns':corr,'native_std_m_s2':a.std(axis=0).tolist(),'projected_std_m_s2':b.std(axis=0).tolist(),'records':observed}
def main():
 plan=json.loads(PLAN.read_text());runs={s:analyze(s,h) for s,h in zip(('01','02'),plan['source_result_sha256'],strict=True)}
 report={'status':'completed_exploratory_world_velocity_derivative_comparison','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'runs':runs,'physical_time_verified':False,'acceleration_axes_registered':False,'training_admissible':False,'actor_qualified':False,'limits':plan['limitations']}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({s:{k:v for k,v in data.items() if k!='records'} for s,data in runs.items()},indent=2))
if __name__=='__main__':main()
