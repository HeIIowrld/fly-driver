"""Offline native acceleration axis check against smoothed speed derivative."""
import base64,hashlib,json
from bisect import bisect_left,bisect_right
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
PLAN=Path('D:/CodexResearchRuns/flybrain-native-accel-axis-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-native-accel-axis-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(suffix,expected):
 root=Path('D:/CodexResearchRuns')/f'flybrain-marker-motion-20261102-{suffix}';result=json.loads((root/'result.json').read_text(encoding='utf8'));p=root/'native-copy-windows.json'
 if sha(root/'result.json')!=expected or result['status']!='completed' or sha(p)!=result['native_copy_sha256']:raise ValueError('source provenance')
 rows=[];ids=set()
 for r in json.loads(p.read_text(encoding='utf8')):
  a=r['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('prefix hash mismatch')
  t=r['copy']['second_copy_end'];s=decode_teacher_physics(raw,t);pid=s.physics.packet_id
  if pid!=a['packet_id']:raise ValueError('packet id mismatch')
  if pid in ids:continue
  ids.add(pid);rows.append((t,s.physics.fields['speedKmh']/3.6,*s.physics.fields['accG']))
 rows=np.array(rows);t=rows[:,0]
 if np.any(np.diff(t)<=0):raise ValueError('nonmonotonic native receipt')
 times=t.tolist();x=[];y=[]
 for i,ti in enumerate(t):
  if rows[i,1]<=5/3.6:continue
  lo=bisect_left(times,ti-.075);hi=bisect_right(times,ti+.075);tt=t[lo:hi]
  if len(tt)<8 or tt[0]>ti-.06 or tt[-1]<ti+.06 or np.max(np.diff(tt))>.02:continue
  delta=tt-ti;speed=rows[lo:hi,1];slope=float(np.dot(delta-delta.mean(),speed-speed.mean())/np.sum((delta-delta.mean())**2));x.append(rows[i,2:5]*9.80665);y.append(slope)
 X=np.asarray(x);Y=np.asarray(y)
 if X.ndim!=2 or X.shape[1]!=3 or len(Y)<20 or not np.isfinite(X).all() or not np.isfinite(Y).all():raise ValueError('insufficient finite moving samples')
 return X,Y,{'unique_native_packets':len(rows),'eligible_windows':len(Y),'min_speed_m_s':float(rows[:,1].min()),'max_speed_m_s':float(rows[:,1].max()),'positive_speed_derivative':int(sum(Y>.1)),'negative_speed_derivative':int(sum(Y<-.1))}
def main():
 plan=json.loads(PLAN.read_text(encoding='utf8'));X,Y,fit=run('01',plan['source_sha256'][0]);Z,W,test=run('02',plan['source_sha256'][1]);corr=[float(np.corrcoef(X[:,i],Y)[0,1]) for i in range(3)]
 if not np.isfinite(corr).all():raise ValueError('constant fit axis')
 axis=int(np.argmax(np.abs(corr)));coef=np.linalg.lstsq(np.column_stack((X[:,axis],np.ones(len(X)))),Y,rcond=None)[0]
 def metrics(A,B):
  pred=A[:,axis]*coef[0]+coef[1];return {'correlation':float(np.corrcoef(A[:,axis],B)[0,1]),'rmse_m_s2':float(np.sqrt(np.mean((pred-B)**2))),'zero_prediction_rmse_m_s2':float(np.sqrt(np.mean(B**2))),'true_mean_m_s2':float(B.mean()),'true_std_m_s2':float(B.std())}
 report={'status':'completed_exploratory_native_acceleration_axis_check','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'fit_axis_correlations':corr,'selected_native_axis_index':axis,'fit_slope':float(coef[0]),'fit_intercept_m_s2':float(coef[1]),'fit_data':fit,'heldout_data':test,'fit_metrics':metrics(X,Y),'heldout_metrics':metrics(Z,W),'physical_time_verified':False,'acceleration_axes_registered':False,'training_admissible':False,'actor_qualified':False,'limitations':plan['limits']}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');np.savez_compressed(OUT/'values.npz',fit_native_m_s2=X,fit_speed_derivative=Y,holdout_native_m_s2=Z,holdout_speed_derivative=W);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
