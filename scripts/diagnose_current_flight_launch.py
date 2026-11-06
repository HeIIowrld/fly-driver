"""Read-only launch diagnosis from original authority bytes and sealed actor arrays."""
import argparse,base64,hashlib,json,configparser,struct
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def unpack(authority):
 raw=base64.b64decode(authority['prefix_base64'],validate=True)
 if hashlib.sha256(raw).hexdigest()!=authority['prefix_sha256']:raise ValueError('Native original hash differs')
 vals=PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]);offset=0;fields={}
 for name,_,n in PHYSICS_LAYOUT:fields[name]=vals[offset] if n==1 else vals[offset:offset+n];offset+=n
 fields['clutch']=struct.unpack_from('<f',raw,364)[0]
 return fields

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=begin_run(a.output,dict(program_sha256=sha256(__file__),seed=None,vehicle_control_sent=False,optimizer_updates=0));cases=[]
 root=Path('runs/student-closed-loop-20261105')
 for name in ['parent40-student-01','candidate40-student-02','parent80-student-04','candidate80-student-02']:
  d=root/name/'run';terminal=read(d/'actuator/actuator-terminal.json');feedback=read(d/'actuator/actuator-feedback.json');events=[v for v in terminal['events'] if v['kind']=='student_request' and v.get('receipt',{}).get('driver_accepted')]
  start=events[0]['receipt']['driver_call_started_perf_counter_seconds'];end=events[-1]['when'];active=[(v,unpack(v['authority'])) for v in feedback if start<=v['first_receipt']<=end]
  u=np.array([[v['receipt']['requested_control'][k] for k in ['steering','throttle','brake']] for v in events]);rpm=[v['rpms'] for _,v in active]
  cases.append(dict(run=str(d),original_feedback_sha256=sha256(d/'actuator/actuator-feedback.json'),active_packets=len({v['packet'] for v,_ in active}),gear_values=sorted({v['gear'] for _,v in active}),auto_shifter_values=sorted({v['autoShifterOn'] for _,v in active}),pit_limiter_values=sorted({v['pitLimiterOn'] for _,v in active}),rpm_range=[min(rpm),max(rpm)],native_gas_brake_mean=np.mean([[v['gas'],v['brake']] for _,v in active],0).tolist(),request_mean=u.mean(0).tolist(),request_range=np.stack([u.min(0),u.max(0)]).tolist(),coactivation_fraction=float(np.mean((u[:,1]>.02)&(u[:,2]>.02))),clutch_feedback_in_saved_authority=True,clutch_range=[min(v['clutch'] for _,v in active),max(v['clutch'] for _,v in active)],clutch_causality='Original420-byte authority includes clutch at364; diagnostic only, do not equate gear with engagement'))
 split=read('runs/thermal-control-improvement-20261104/newhot-cohort-01.json');datasets=[]
 for ep in split['episodes']:
  fn=Path(ep['data'])/'actor-only.npz'
  if sha256(fn)!=ep['arrays_sha256']:raise ValueError('Original actor data changed')
  with np.load(fn,allow_pickle=False) as z:
   speed=z['body'][:,16];valid=z['body_valid'][:,16];fit=z['fit_mask'];lab=z['label_valid'];rows=[]
   for lo,hi in [(0,.5),(.5,2),(2,5),(5,13),(13,1000)]:
    m=valid&(speed>=lo)&(speed<hi);ml=m&lab
    rows.append(dict(speed_range_m_s=[lo,hi],all=int(m.sum()),label_valid=int(ml.sum()),fit=int((m&fit).sum()),label_mean=None if not ml.any() else z['labels'][ml].mean(0).tolist(),label_pedal_coactivation=None if not ml.any() else float(np.mean((z['labels'][ml,1]>.02)&(z['labels'][ml,2]>.02)))))
   datasets.append(dict(role=ep['role'],data=ep['data'],arrays_sha256=ep['arrays_sha256'],events=len(speed),all_speed_range=[float(speed.min()),float(speed.max())],fit_speed_range=[float(speed[fit].min()),float(speed[fit].max())],bins=rows,first_fit=int(np.flatnonzero(fit)[0]),first_valid_label=int(np.flatnonzero(lab)[0])))
 axis=read('runs/thermal-driving-20261103/steering-calibration-02/axis-config.json') if Path('runs/thermal-driving-20261103/steering-calibration-02/axis-config.json').exists() else read('runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json')
 directory=Path(axis['profile_path']).parent;settings={}
 for name in ['controls.ini','assists.ini']:
  fn=directory/name;cp=configparser.ConfigParser(strict=False);cp.read(fn,encoding='utf-8-sig');settings[name]=dict(path=str(fn),sha256=sha256(fn),selected={section:dict(cp[section]) for section in cp.sections() if section in ['ASSISTS','CLUTCH','GEAR_UP','GEAR_DOWN']})
 result=dict(status='completed_readonly_current_flight_launch_diagnosis',cases=cases,datasets=datasets,actual_settings=settings,vehicle_control_sent=False,optimizer_updates=0,ui_actions_sent=0)
 write_json(out/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
