"""Evaluate fixed forecast and feature geometry on a varied synthetic stimulus."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from scripts.audit_motion_increment_geometry import stats,summarize
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=B/'20261101-varied-motion-causal-timing';r=read(root/'result.json');man=read(root/'manifest.json')['config'];f=root/'responses.npz'
 if r['status']!='completed_causal_prior_frame_replay' or r['responses_sha256']!=sha256(f) or man['program_sha256']!=sha256('scripts/replay_varied_motion_causal_timing.py'):raise ValueError('stimulus lineage')
 with np.load(f,allow_pickle=False) as z:retina=z['retina'].copy();motion=z['motion'].copy();times=z['times'].copy()
 if retina.shape!=(3,41,4,721) or motion.shape!=(3,41,4,8,721) or not np.isfinite(retina).all() or not np.isfinite(motion).all() or not np.allclose(times,np.arange(41)*.04,atol=1e-12) or not np.all(retina[:,:11]==retina[:,:1]):raise ValueError('incomplete stimulus')
 torch.set_num_threads(1);spec=read('configs/flight-circuit-visual-v6-20261029.json')
 with load_array(spec['scene_registration_file'],allow_pickle=False) as z:pool=FlightScenePool(z['image_uv'],spec['scene_basis'])
 x=np.concatenate([retina[:,:,:,None],motion],axis=3)
 with torch.no_grad():scene=pool(torch.tensor(x.reshape(-1,4,9,721),dtype=torch.float32)).numpy().reshape(3,41,4,9,16)[:,:,:,1:].astype('float64')
 mask=np.arange(39)>=9;parts=[stats(scene[i]) for i in (1,2)];geometry=summarize(*(np.concatenate([part[j][mask] for part in parts]) for j in range(3)))
 pp=np.concatenate([(scene[i,1:-1]-scene[i,:-2])[mask] for i in (1,2)]);nn=np.concatenate([(scene[i,2:]-scene[i,1:-1])[mask] for i in (1,2)]);h=float(np.square(nn).mean());models={};hashes={}
 for name,folder in [('original','20261101-causal-motion-forecast-synthetic-fit'),('reversal','20261101-reversing-yaw-forecast-synthetic-fit')]:
  root=B/folder;receipt=read(root/'result.json');ck=root/'motion-forecast.pt'
  if sha256(ck)!=receipt['checkpoint_sha256'] or receipt['actor_qualified'] is not False:raise ValueError('forecast lineage')
  payload=torch.load(ck,map_location='cpu',weights_only=True);g=payload['model']['raw_gain'].tanh().numpy().reshape(1,1,8,1);m=float(np.square(pp*g-nn).mean());models[name]=dict(mse=m,relative_change=m/h-1);hashes[name]=sha256(ck)
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),stimulus_sha256=sha256(f),checkpoint_sha256=hashes,scope='diagnostic synthetic stimulus, no fit'))
 out=dict(status='completed_causally_timed_varied_synthetic_assessment',geometry=geometry,hold_mse=h,models=models,interpretation='Single diagnostic synthetic texture and engineered motion, selected after prior results; not independent validation or physical physiology.',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 write_json(run/'result.json',out);print(json.dumps(out,allow_nan=False))
if __name__=='__main__':main()
