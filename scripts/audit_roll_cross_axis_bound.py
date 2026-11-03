"""Bound possible Euler cross-axis roll-rate coupling in post-hoc windows."""
import argparse,base64,hashlib,json
from pathlib import Path
from bisect import bisect_left,bisect_right
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=B/'20261101-historical-rotation-window-trace';tr=read(root/'result.json');rr=B/'20261101-native-renderer-roll-chain';res=read(rr/'result.json');
 if sha256(root/'comparisons.npz')!=tr['output_sha256'] or sha256(rr/'comparison.npz')!=res['output_sha256']:raise ValueError('original comparison hashes required')
 reports=[];arrays={}
 with np.load(root/'comparisons.npz',allow_pickle=False) as z,np.load(rr/'comparison.npz',allow_pickle=False) as c:
  for i,entry in enumerate(tr['reports']):
   ef=Path(entry['recording'])/'events.jsonl'
   if sha256(ef)!=entry['events_sha256']:raise ValueError('source event mismatch')
   unique={}
   for line in ef.open(encoding='utf8'):
    e=json.loads(line);kind=e.get('kind');engine=kind is None and all(k in e for k in ('native','camera','decision_monotonic_seconds'))
    if kind not in ('raw_pair','expert_raw_native') and not engine:continue
    n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True);h=hashlib.sha256(raw).hexdigest()
    if h!=n['prefix_sha256']:raise ValueError('packet hash mismatch')
    s=decode_virtual_body_prefix(raw,float(n['received_monotonic_seconds']));pid=s.packet_id
    if pid!=n['packet_id'] or (pid in unique and unique[pid]['hash']!=h):raise ValueError('packet inconsistency')
    unique.setdefault(pid,{'time':s.received_monotonic_seconds,'hash':h,'pitch':s.pitch_native,'q':s.angular_native[0],'r':s.angular_native[1]})
   values=sorted(unique.values(),key=lambda x:x['time']);times=np.array([x['time'] for x in values]);pitch=np.array([x['pitch'] for x in values]);q=np.array([x['q'] for x in values]);r=np.array([x['r'] for x in values]);meta=z[f'window_metadata_{i}'];chain=c[f'windows_{i}'];np.testing.assert_array_equal(meta[:,0],chain[:,0]);early=meta[:,0]<=25;records=[]
   for j in np.flatnonzero(early):
    lo,hi=meta[j,1:3];left=bisect_right(times,lo)-1;right=bisect_left(times,hi)
    if left<0 or right>=len(times):raise ValueError('bracket absent')
    angle=float(np.max(np.abs(pitch[left:right+1])));qmax=float(np.max(np.abs(q[left:right+1])));rmax=float(np.max(np.abs(r[left:right+1])))
    if angle>=1.4:raise ValueError('Euler near singularity')
    bound=np.tan(angle)*(qmax+rmax);difference=abs(float(chain[j,1]-chain[j,2]));records.append([meta[j,0],bound,difference])
   arr=np.asarray(records);arrays[f'early_{i}']=arr;reports.append({'recording':entry['recording'],'windows':len(arr),'max_coupling_bound':float(arr[:,1].max()),'median_coupling_bound':float(np.median(arr[:,1])),'gyro_minus_angle_rmse':float(np.sqrt(np.mean(arr[:,2]**2))),'windows_exceeding_bound':int(np.sum(arr[:,2]>arr[:,1]))})
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),window_sha256=tr['output_sha256'],chain_sha256=res['output_sha256'],assumption='ZYX Euler with mapped q/r, interpolated host timestamps; bound tan(abs pitch)*(abs q + abs r)',posthoc=True,optimizer_updates=0));np.savez_compressed(run/'bounds.npz',**arrays);out={'status':'completed_roll_cross_axis_bound_diagnostic','reports':reports,'output_sha256':sha256(run/'bounds.npz'),'axis_validity_changed':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'conditional on assumed ZYX axes, sampled rates and interpolation; shared simulator and host timestamps not physically synchronized; not proof of sensor error'};write_json(run/'result.json',out);print(json.dumps(out))
if __name__=='__main__':main()
