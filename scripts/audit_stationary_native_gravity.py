"""Inspect native acceleration in consistently stationary diagnostic windows."""
import base64,hashlib,json
from bisect import bisect_left,bisect_right
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.virtual_body import decode_virtual_body_prefix
PLAN=Path('D:/CodexResearchRuns/flybrain-native-stationary-gravity-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-native-stationary-gravity-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(suffix,expected):
 root=Path('D:/CodexResearchRuns')/f'flybrain-marker-motion-20261102-{suffix}';r=json.loads((root/'result.json').read_text(encoding='utf8'));p=root/'native-copy-windows.json'
 if sha(root/'result.json')!=expected or r['status']!='completed' or sha(p)!=r['native_copy_sha256']:raise ValueError('source mismatch')
 rows=[];seen=set()
 for row in json.loads(p.read_text(encoding='utf8')):
  a=row['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('prefix mismatch')
  t=row['copy']['second_copy_end'];sample=decode_teacher_physics(raw,t)
  if sample.physics.packet_id!=a['packet_id']:raise ValueError('id mismatch')
  if sample.physics.packet_id in seen:continue
  seen.add(sample.physics.packet_id);v=sample.physics.fields;b=decode_virtual_body_prefix(raw[:PHYSICS_PREFIX.size],t)
  rows.append([t,v['speedKmh'],*v['localAngularVel'],*v['accG'],b.pitch_native,b.roll_native])
 data=np.asarray(rows);times=data[:,0]
 if np.any(np.diff(times)<=0):raise ValueError('nonmonotonic receipt')
 seq=times.tolist();good=(data[:,1]<.1)&(np.max(np.abs(data[:,2:5]),axis=1)<.01);indices=[]
 for i,t in enumerate(times):
  lo=bisect_left(seq,t-.5);hi=bisect_right(seq,t+.5);ts=times[lo:hi]
  if len(ts)<50 or ts[0]>t-.48 or ts[-1]<t+.48 or np.max(np.diff(ts))>.02 or not np.all(good[lo:hi]):continue
  indices.append(i)
 if not indices:raise ValueError('no qualifying stationary windows')
 sample=data[indices];acc=sample[:,5:8];norm=np.linalg.norm(acc,axis=1)
 return {'unique_packets':len(data),'qualifying_packets':len(sample),'stationary_coverage_seconds':float(sample[-1,0]-sample[0,0]),'pitch_range_rad':[float(sample[:,8].min()),float(sample[:,8].max())],'roll_range_rad':[float(sample[:,9].min()),float(sample[:,9].max())],'acceleration_norm_g':{'median':float(np.median(norm)),'p95':float(np.percentile(norm,95)),'max':float(norm.max())},'axis_mean_g':acc.mean(axis=0).tolist(),'axis_median_g':np.median(acc,axis=0).tolist(),'zero_vector_fraction':float(np.mean(norm<1e-6)),'source_sha256':sha(p)}
def main():
 plan=json.loads(PLAN.read_text(encoding='utf8'));runs={s:analyze(s,h) for s,h in zip(('01','02'),plan['source_result_sha256'],strict=True)}
 report={'status':'completed_stationary_native_acceleration_diagnostic','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'runs':runs,'acceleration_axes_registered':False,'physical_time_verified':False,'training_admissible':False,'actor_qualified':False,'limits':plan['limitations']}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
