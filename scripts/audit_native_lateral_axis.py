"""Compare native accelerometer axes to a provisional lateral-motion proxy."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
BASE=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');PLAN=Path('D:/CodexResearchRuns/flybrain-native-lateral-axis-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-native-lateral-axis-20261102-01')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(role,expected):
 root=BASE/f'20261031-pace-{role}-native-copy-candidate-00/recording';result=json.loads((root/'result.json').read_text(encoding='utf8'));file=root/'native-copy-history.json';meta=result['native_copy_history']
 if sha(root/'result.json')!=expected or result['status']!='completed' or result['training_admissible'] is not False or meta['fit_permission_granted'] is not False or sha(file)!=meta['sha256']:raise ValueError('unqualified diagnostic source required')
 records=json.loads(file.read_text(encoding='utf8'))
 if len(records)!=meta['rows']:raise ValueError('native copy incomplete')
 seen={};rows=[]
 for row in records:
  a=row['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True);digest=hashlib.sha256(raw).hexdigest()
  if digest!=a['prefix_sha256']:raise ValueError('prefix hash mismatch')
  sample=decode_teacher_physics(raw,a['received_monotonic_seconds']);pid=sample.physics.packet_id
  if pid!=a['packet_id']:raise ValueError('packet ID mismatch')
  if pid in seen:
   if seen[pid]!=digest:raise ValueError('same native packet changed')
   continue
  seen[pid]=digest;v=sample.physics.fields;speed=v['speedKmh']/3.6;yaw=v['localAngularVel'][1]
  if speed>=5 and abs(yaw)>=.02:rows.append([speed,yaw,*v['accG']])
 data=np.asarray(rows,dtype=float)
 if data.ndim!=2 or data.shape[1]!=5 or len(data)<50 or not np.isfinite(data).all():raise ValueError('insufficient eligible finite samples')
 return data,{'unique_packets':len(seen),'eligible_packets':len(data),'positive_yaw_packets':int(sum(data[:,1]>0)),'negative_yaw_packets':int(sum(data[:,1]<0)),'max_speed_m_s':float(data[:,0].max()),'min_yaw_rad_s':float(data[:,1].min()),'max_yaw_rad_s':float(data[:,1].max()),'source_history_sha256':sha(file)}
def main():
 plan=json.loads(PLAN.read_text(encoding='utf8'));a,am=read('train',plan['source_result_sha256'][0]);b,bm=read('development',plan['source_result_sha256'][1])
 def corr(d,i):return float(np.corrcoef(d[:,0]*d[:,1],d[:,2+i]*9.80665)[0,1])
 cs=[corr(a,i) for i in range(3)]
 if not np.isfinite(cs).all():raise ValueError('constant fit axis')
 idx=int(np.argmax(np.abs(cs)))
 def metrics(d):
  proxy=d[:,0]*d[:,1];acc=d[:,2+idx]*9.80665;sign=1 if cs[idx]>=0 else -1
  return {'correlation':float(np.corrcoef(proxy,acc)[0,1]),'same_signed_fraction':float(np.mean((proxy*acc*sign)>0)),'proxy_range_m_s2':[float(proxy.min()),float(proxy.max())],'axis_range_m_s2':[float(acc.min()),float(acc.max())]}
 report={'status':'completed_exploratory_native_lateral_axis','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'first_recording_axis_correlations':cs,'selected_native_index':idx,'first_recording':am,'second_recording':bm,'first_metrics':metrics(a),'second_metrics':metrics(b),'acceleration_axes_registered':False,'physical_time_verified':False,'training_admissible':False,'actor_qualified':False,'limitations':plan['limits']}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
