"""Compare motion feature increment geometry, without fitting to real histories."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from scripts.train_motion_forecast_reversal_synthetic import load_reversal
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def stats(x):
 prev=x[1:-1]-x[:-2];nxt=x[2:]-x[1:-1]
 pp=np.sum(prev*prev,axis=(1,2,3));nn=np.sum(nxt*nxt,axis=(1,2,3));dot=np.sum(prev*nxt,axis=(1,2,3)); denom=np.sqrt(pp*nn)
 cos=np.divide(dot,denom,out=np.zeros_like(dot),where=denom>1e-15)
 return pp,nn,cos
def summarize(pp,nn,cos):
 return dict(windows=int(len(cos)),previous_rms=float(np.sqrt(pp.mean()/512)),next_rms=float(np.sqrt(nn.mean()/512)), cosine_median=float(np.median(cos)),cosine_mean=float(cos.mean()),aligned_fraction=float((cos>.1).mean()),reversed_fraction=float((cos<-.1).mean()),weak_fraction=float((abs(cos)<=.1).mean()), cosine_p10=float(np.percentile(cos,10)),cosine_p90=float(np.percentile(cos,90)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 spec=read('configs/flight-circuit-visual-v6-20261029.json');plan=read(B/'20261101-reversing-yaw-forecast-plan/plan.json')
 with load_array(spec['scene_registration_file'],allow_pickle=False) as z:pool=FlightScenePool(z['image_uv'],spec['scene_basis'])
 torch.set_num_threads(1);synthetic_root=B/'20261101-flyvis-reversing-yaw-879403'
 syn=load_reversal(synthetic_root,879403,.13,pool,plan['generator_sha256']).numpy().reshape(3,41,4,9,16)[:,:,:,1:].astype('float64')
 root=B/'20261029-t1434-body21-flight-cache'; manifest=read(root/'dataset.json');result=read(root/'result.json')
 if sha256(root/'dataset.json')!=result['dataset_manifest_sha256']:raise ValueError('manifest mismatch')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),dataset_sha256=sha256(root/'dataset.json'),synthetic_sha256=sha256(synthetic_root/'responses.npz'),scope='posthoc descriptive comparison, no fitting'))
 vals=[[],[],[]];eps=[]
 for ep in manifest['episodes']:
  if sha256(resolve_payload_path(root/ep['file']))!=ep['sha256'] or read(Path(ep['original']['recording'])/'result.json').get('training_admissible') is not False:raise ValueError('source')
  with load_array(root/ep['file'],allow_pickle=False) as z:scene=z['scene_features'].copy();clock=z['observation_time'].copy()
  if scene.shape!=(len(clock),576) or not np.isfinite(scene).all() or not np.isfinite(clock).all():raise ValueError('scene')
  dt=np.diff(clock);eligible=(dt[:-1]>=.036)&(dt[:-1]<=.044)&(dt[1:]>=.036)&(dt[1:]<=.044)
  p0,n0,c=stats(scene.reshape(-1,4,9,16)[:,:,1:].astype('float64')); s=[v[eligible] for v in (p0,n0,c)]
  for dest,v in zip(vals,s):dest.append(v)
  eps.append(dict(file=ep['file'],**summarize(*s)))
 real=summarize(*(np.concatenate(x) for x in vals)); sections={}
 for name,mask in [('all',np.ones(39,dtype=bool)),('motion',np.arange(39)>=9),('motion_away_from_reversal',(np.arange(39)>=9)&~np.isin(np.arange(39),[23,24,25,26])),('reversal_neighborhood',np.isin(np.arange(39),[23,24,25,26]))]:
  parts=[stats(syn[i]) for i in (1,2)]; sections[name]=summarize(*(np.concatenate([part[j][mask] for part in parts]) for j in range(3)))
 record=dict(status='completed_increment_geometry_comparison',real_historical=real,real_episode=eps,previously_seen_synthetic_holdout=sections,interpretation='Posthoc descriptive features only; real examples temporally correlated and source fitting prohibited; synthetic phase is ground truth used only for reporting, not model input.',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False,physical_exposure_verified=False)
 write_json(run/'result.json',record);print(json.dumps(record,allow_nan=False))
if __name__=='__main__':main()
