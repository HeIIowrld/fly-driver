"""Exploratory load-order crosscheck from retained AC native packets; no admission."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=read(a.plan);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),diagnostic_only=True,actor=False));result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  reports=[]
  for src in plan['sources']:
   root=Path(src['root']);resultfile=root/'result.json';datafile=root/src['file'];receipt=read(resultfile)
   if receipt['status']!='completed' or sha256(resultfile)!=src['result_sha256'] or sha256(datafile)!=src['data_sha256']:raise ValueError('exact completed source required')
   if src['kind']=='marker' and receipt['native_copy_sha256']!=src['data_sha256']:raise ValueError('marker copy receipt mismatch')
   raw=read(datafile);seen=set();rows=[];invalid=0;duplicate=0
   for item in raw:
    item=item['authority'] if src['kind']=='marker' else item
    data=base64.b64decode(item['prefix_base64'],validate=True)
    if hashlib.sha256(data).hexdigest()!=item['prefix_sha256']:raise ValueError('native packet digest mismatch')
    time=float(item['received_monotonic_seconds']);s=(decode_teacher_physics(data,time).physics if src['kind']=='marker' else decode_physics_prefix(data,time));pid=s.packet_id
    if pid!=item['packet_id']:raise ValueError('packet identity mismatch')
    if pid in seen:duplicate+=1;continue
    seen.add(pid);f=s.fields;load=np.asarray(f['wheelLoad'],dtype=float);acc=np.asarray(f['accG'],dtype=float);speed=float(f['speedKmh'])/3.6
    if not np.isfinite(load).all() or not np.isfinite(acc).all() or not np.isfinite(speed) or np.any(load<0) or load.sum()<=0:invalid+=1;continue
    load=load/load.sum();rows.append([time,speed,*acc,*load])
   x=np.asarray(rows);mask=x[:,1]>=plan['minimum_speed_m_s'];x=x[mask]
   if len(x)<20:raise ValueError('insufficient moving unique native packets')
   lat=x[:,6]+x[:,8]-x[:,5]-x[:,7];front=x[:,5]+x[:,6]-x[:,7]-x[:,8];axis0=x[:,2];axis2=x[:,4]
   def corr(u,v):
    return float(np.corrcoef(u,v)[0,1]) if np.std(u)>1e-8 and np.std(v)>1e-8 else None
   reports.append(dict(name=src['name'],unique_moving_packets=len(x),duplicate_packets=duplicate,invalid_packets=invalid,seconds=float(x[-1,0]-x[0,0]),speed_range=[float(x[:,1].min()),float(x[:,1].max())],left_right_contrast_std=float(np.std(lat)),front_rear_contrast_std=float(np.std(front)),axis0_std_g=float(np.std(axis0)),axis2_std_g=float(np.std(axis2)),lateral_vs_axis0_correlation=corr(lat,axis0),longitudinal_vs_axis2_correlation=corr(front,axis2),lateral_vs_axis2_correlation=corr(lat,axis2),longitudinal_vs_axis0_correlation=corr(front,axis0)))
  result.update(status='completed_exploratory_native_wheel_order_audit',reports=reports,physical_time_verified=False,wheel_order_registered=False,training_admissible=False,interpretation='Hypothesized native index order FL FR RL RR, acceleration axis/sign hypotheses; same-packet correlations only. Samples time-correlated, same simulator/car may be repeated; no intervention or independent wheel-contact ground truth.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_exploratory_native_wheel_order_audit':raise SystemExit(1)
if __name__=='__main__':main()
