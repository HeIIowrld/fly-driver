"""Locate early roll mismatch relative to original launch telemetry and controls."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.ac_native_physics import decode_physics_prefix
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();trace=B/'20261101-historical-rotation-window-trace';tr=read(trace/'result.json');fn=trace/'comparisons.npz';other=B/'20261101-causal-available-integrated-rotation';orr=read(other/'result.json');of=other/'angular-comparison.npz'
 if sha256(fn)!=tr['output_sha256'] or sha256(of)!=orr['comparison_sha256']:raise ValueError('exact original comparisons required')
 reports=[];arrays={}
 with np.load(fn,allow_pickle=False) as z:
  for i,entry in enumerate(tr['reports']):
   ef=Path(entry['recording'])/'events.jsonl'
   if sha256(ef)!=entry['events_sha256']:raise ValueError('original events mismatch')
   meta=z[f'window_metadata_{i}'];m=meta[:,0]<=25;lo=float(meta[m,1].min());hi=float(meta[m,2].max());unique={};requests=[]
   for line in ef.open(encoding='utf8'):
    e=json.loads(line);kind=e.get('kind');t=e.get('host_monotonic_seconds')
    if kind in ('open','student_request','close') and t is not None and lo<=t<=hi:requests.append({'kind':kind,'offset_s':float(t-lo)})
    if kind not in ('raw_pair','expert_raw_native'):continue
    n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
    if h!=n['prefix_sha256']:raise ValueError('native hash mismatch')
    time=float(n['received_monotonic_seconds']);s=decode_physics_prefix(raw,time)
    if s.packet_id!=n['packet_id'] or (s.packet_id in unique and unique[s.packet_id]['hash']!=h):raise ValueError('packet identity mismatch')
    if lo<=time<=hi:unique.setdefault(s.packet_id,{'time':time,'hash':h,'fields':s.fields})
   rows=sorted(unique.values(),key=lambda x:x['time']);values=np.array([[x['time']-lo,x['fields']['speedKmh'],x['fields']['gas'],x['fields']['brake'],x['fields']['steerAngle'],*x['fields']['localAngularVel']] for x in rows]);
   if len(values)<3 or not np.isfinite(values).all():raise ValueError('original finite early native samples required')
   peak=values[np.argmax(np.abs(values[:,7]))];arrays[f'early_native_{i}']=values;reports.append({'recording':entry['recording'],'capture_early_host_duration_s':hi-lo,'native_packets':len(values),'speed_kmh_minmax':[float(values[:,1].min()),float(values[:,1].max())],'gas_minmax':[float(values[:,2].min()),float(values[:,2].max())],'brake_minmax':[float(values[:,3].min()),float(values[:,3].max())],'native_steer_minmax':[float(values[:,4].min()),float(values[:,4].max())],'peak_abs_native_roll_rate':float(peak[7]),'peak_offset_s':float(peak[0]),'speed_at_peak_kmh':float(peak[1]),'recorded_student_request_count':sum(x['kind']=='student_request' for x in requests),'request_kinds':sorted(set(x['kind'] for x in requests))})
 with np.load(of,allow_pickle=False) as z:
  native=z['native'];render=z['renderer'];residual=native[:,0]-render[:,0];comparison={'windows':len(residual),'roll_residual_max_abs':float(np.max(np.abs(residual))),'roll_residual_first15_max_abs':float(np.max(np.abs(residual[:15]))),'native_roll_range':float(np.ptp(native[:,0])),'different_probe_and_window_method':True}
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),window_source_sha256=tr['output_sha256'],independent_probe_sha256=orr['comparison_sha256'],early_selection='previously identified capture <=25',optimizer_updates=0));np.savez_compressed(run/'early-native.npz',**arrays)
 result={'status':'completed_early_roll_launch_context','reports':reports,'other_low_speed_probe':comparison,'output_sha256':sha256(run/'early-native.npz'),'training_admissible':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'Temporal co-occurrence only, previous recordings and differing camera windows; source timestamps and exposure not physically synchronized. Requests are not proof of applied control or causation.'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
