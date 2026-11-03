"""Compare native roll-angle derivative to native gyro and renderer in original windows."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();trace=B/'20261101-historical-rotation-window-trace';tr=read(trace/'result.json');f=trace/'comparisons.npz'
 if tr['status']!='completed_historical_integrated_rotation_window_trace' or sha256(f)!=tr['output_sha256']:raise ValueError('exact prior window traces required')
 reports=[];arrays={}
 with np.load(f,allow_pickle=False) as z:
  for i,entry in enumerate(tr['reports']):
   root=Path(entry['recording']);events=root/'events.jsonl'
   if sha256(events)!=entry['events_sha256']:raise ValueError('original event hash mismatch')
   unique={}
   with events.open(encoding='utf8') as stream:
    for line in stream:
     e=json.loads(line);kind=e.get('kind');engine=kind is None and all(k in e for k in ('native','camera','decision_monotonic_seconds'))
     if kind not in ('raw_pair','expert_raw_native') and not engine:continue
     n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
     if h!=n['prefix_sha256']:raise ValueError('original prefix mismatch')
     sample=decode_virtual_body_prefix(raw,float(n['received_monotonic_seconds']));pid=sample.packet_id
     if pid!=n['packet_id'] or (pid in unique and unique[pid]['hash']!=h):raise ValueError('native packet inconsistency')
     unique.setdefault(pid,{'time':sample.received_monotonic_seconds,'hash':h,'roll':sample.roll_native,'pitch':sample.pitch_native})
   packets=sorted(unique.values(),key=lambda x:x['time']);times=np.array([x['time'] for x in packets]);roll=np.array([x['roll'] for x in packets]);pitch=np.array([x['pitch'] for x in packets])
   if not np.all(np.diff(times)>0):raise ValueError('ordered original receipt times required')
   x=z[f'native_{i}'];y=z[f'renderer_{i}'];meta=z[f'window_metadata_{i}'];lo=meta[:,1];hi=meta[:,2]
   if not ((lo>=times[0]).all() and (hi<=times[-1]).all() and np.all(hi>lo)):raise ValueError('native bracket missing')
   # Small-tilt approximation only; host bounds are not physical event times.
   native_roll_diff=-(np.interp(hi,times,roll)-np.interp(lo,times,roll))/(hi-lo);native_pitch_diff=-(np.interp(hi,times,pitch)-np.interp(lo,times,pitch))/(hi-lo)
   values=np.column_stack([meta[:,0],native_roll_diff,x[:,0],y[:,0],native_pitch_diff,x[:,1],y[:,1]]);arrays[f'windows_{i}']=values
   groups={}
   for label,mask in [('early',meta[:,0]<=25),('later',meta[:,0]>=30)]:
    if not mask.any():continue
    groups[label]={'samples':int(mask.sum()),'native_gyro_vs_native_angle_rmse':float(np.sqrt(np.mean((x[mask,0]-native_roll_diff[mask])**2))),'native_angle_vs_renderer_rmse':float(np.sqrt(np.mean((native_roll_diff[mask]-y[mask,0])**2))),'gyro_vs_renderer_rmse':float(np.sqrt(np.mean((x[mask,0]-y[mask,0])**2))),'mean_native_gyro':float(np.mean(x[mask,0])),'mean_native_angle_derivative':float(np.mean(native_roll_diff[mask])),'mean_renderer_rate':float(np.mean(y[mask,0]))}
   reports.append({'recording':entry['recording'],'windows':len(x),'groups':groups,'events_sha256':entry['events_sha256']})
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),window_trace_sha256=tr['output_sha256'],native_pose_derivative='-roll difference / host interval, small-angle approximation',early_cutoff=25,later_start=30,posthoc_selection=True,optimizer_updates=0));np.savez_compressed(run/'comparison.npz',**arrays)
 result={'status':'completed_native_renderer_roll_chain_exploration','reports':reports,'output_sha256':sha256(run/'comparison.npz'),'training_admissible':False,'axis_validity_changed':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'Euler roll derivative is only a small-tilt approximation to body roll rate; host lower bounds not physical exposure, correlated windows and same simulator, post-hoc selection.'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
