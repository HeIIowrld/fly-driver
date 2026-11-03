"""Exploratory native attitude finite differences vs native angular rates; no axis admission."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=B/'20261101-low-speed-fourview-stationary-prefix-probe';r=read(root/'result.json');audit=read(B/'20261101-low-speed-fourview-stationary-prefix-body-audit/result.json');host=read(B/'20261101-low-speed-fourview-stationary-prefix-host-audit/manifest.json')['config'];events=root/'events.jsonl'
 if r['status']!='completed_low_speed_pulse_diagnostic' or r['training_admissible'] is not False or audit['status']!='completed_low_speed_raw_audit' or sha256(events)!=host['events_sha256']:raise ValueError('exact diagnostic source required')
 rows=[];last=None
 with events.open(encoding='utf8') as f:
  for line in f:
   e=json.loads(line)
   if e.get('kind')!='native':continue
   n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
   if h!=n['prefix_sha256']:raise ValueError('native hash mismatch')
   sample=decode_virtual_body_prefix(raw,float(n['received_monotonic_seconds']))
   if sample.packet_id!=n['packet_id']:raise ValueError('packet identity mismatch')
   if last is not None and sample.packet_id<last['packet']:raise ValueError('packet reset')
   if last is not None and sample.packet_id==last['packet']:
    if h!=last['hash']:raise ValueError('duplicate bytes changed')
    continue
   row={'packet':sample.packet_id,'hash':h,'time':sample.received_monotonic_seconds,'pitch':sample.pitch_native,'roll':sample.roll_native,'angular':sample.angular_native};rows.append(row);last=row
 times=np.array([x['time'] for x in rows]);pitch=np.array([x['pitch'] for x in rows]);roll=np.array([x['roll'] for x in rows]);angular=np.array([x['angular'] for x in rows])
 if len(rows)<5 or not np.all(np.diff(times)>0) or not np.isfinite(angular).all():raise ValueError('strict original packet order required')
 dt=times[2:]-times[:-2];keep=(dt>.02)&(dt<.2);dt=dt[keep];t0=times[:-2][keep];tm=times[1:-1][keep];t1=times[2:][keep]
 derivative=np.column_stack([-(roll[2:]-roll[:-2])[keep]/dt,-(pitch[2:]-pitch[:-2])[keep]/dt]);mapped=angular[:,[2,0]];center=mapped[1:-1][keep];weighted=((mapped[:-2][keep]+mapped[1:-1][keep])*(tm-t0)[:,None]+(mapped[1:-1][keep]+mapped[2:][keep])*(t1-tm)[:,None])/(2*dt[:,None])
 reports={}
 for i,name in enumerate(('roll','pitch')):
  reports[name]={}
  for mode,x in [('center',center[:,i]),('time_weighted',weighted[:,i])]:
   y=derivative[:,i];slope,offset=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0];reports[name][mode]={'samples':len(x),'native_range':float(np.ptp(x)),'slope':float(slope),'correlation':float(np.corrcoef(x,y)[0,1]),'rmse':float(np.sqrt(np.mean((x-y)**2)))}
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(events),native_time='host first receipt, not simulator event time',axis_hypothesis={'roll_native_index':2,'pitch_native_index':0,'attitude_derivatives':'-native_roll, -native_pitch; small-angle approximation'},optimizer_updates=0));np.savez_compressed(run/'comparison.npz',dt=dt,derivative=derivative,center=center,weighted=weighted)
 result={'status':'completed_native_attitude_rate_exploration','reports':reports,'packets':len(rows),'comparison_sha256':sha256(run/'comparison.npz'),'source_clock_verified':False,'physical_axis_verified':False,'axis_validity_changed':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'Same native sensor source is not independent; Euler derivatives approximate body rates only for small tilt/cross-axis motion, host timing not physics time, windows overlap.'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
