"""Fit the bounded forecast only on predeclared causally replayed synthetic scenes."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_motion_forecast import CausalMotionForecast
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(seed,amp,pool,plan):
 d=B/f'20261101-flyvis-varied-causal-{seed}';r=read(d/'result.json');cfg=read(d/'manifest.json')['config'];f=d/'responses.npz'
 if r['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=r['responses_sha256'] or cfg['program_sha256']!=plan['generator_sha256'] or cfg['texture_seed']!=seed or r['cases']!=[0.,-amp,amp] or cfg['temporal_policy']!='initial grey steady; previous frame held until current decision':raise ValueError('source lineage')
 with np.load(f,allow_pickle=False) as z:retina=z['retina'].copy();motion=z['motion'].copy();times=z['times'].copy()
 if retina.shape!=(3,41,4,721) or motion.shape!=(3,41,4,8,721) or not np.isfinite(retina).all() or not np.isfinite(motion).all() or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12) or not np.all(retina[:,:11]==retina[:,:1]):raise ValueError('source invalid')
 x=np.concatenate([retina[:,:,:,None],motion],axis=3)
 with torch.no_grad():scene=pool(torch.tensor(x.reshape(-1,4,9,721),dtype=torch.float32)).reshape(3,41,576)
 return scene,sha256(f)
def arrays(model,scene):
 a=scene[1:,:-2].reshape(-1,576);b=scene[1:,1:-1].reshape(-1,576);target=scene[1:,2:].reshape(2,39,4,9,16)[:,:,:,1:]
 pred=model(b,a,.04).reshape(2,39,4,9,16)[:,:,:,1:]
 return pred[:,9:],target[:,9:]
def mse(model,scene):
 with torch.no_grad():p,t=arrays(model,scene);return float((p-t).square().mean())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();planpath=B/'20261101-causal-varied-forecast-plan/plan.json';plan=read(planpath)
 if sha256('scripts/replay_flyvis_varied_motion_causal.py')!=plan['generator_sha256'] or sha256('src/flybrain_drive/flight_motion_forecast.py')!=plan['component_sha256']:raise ValueError('plan mismatch')
 torch.set_num_threads(1);torch.manual_seed(863009);spec=read('configs/flight-circuit-visual-v6-20261029.json')
 with load_array(spec['scene_registration_file'],allow_pickle=False) as z:pool=FlightScenePool(z['image_uv'],spec['scene_basis'])
 fit=[load(i,.23,pool,plan) for i in plan['fit_seeds']];run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),sources=[h for _,h in fit],seed=863009,actor=False,real_history_used_for_fit=False))
 model=CausalMotionForecast();optim=torch.optim.Adam(model.parameters(),lr=.03);losses=[]
 for _ in range(200):
  optim.zero_grad(set_to_none=True);loss=torch.stack([(lambda pt:(pt[0]-pt[1]).square().mean())(arrays(model,x)) for x,_ in fit]).mean()
  if not torch.isfinite(loss):raise ValueError('nonfinite loss')
  loss.backward()
  if model.raw_gain.grad is None or not torch.isfinite(model.raw_gain.grad).all():raise ValueError('nonfinite gradient')
  optim.step();losses.append(float(loss.detach()))
 valid,h=load(plan['validation_seed'],.13,pool,plan);hold=CausalMotionForecast()
 comparison={'hold_mse':mse(hold,valid),'new_mse':mse(model,valid)}
 for name,folder in [('original','20261101-causal-motion-forecast-synthetic-fit'),('reversal','20261101-reversing-yaw-forecast-synthetic-fit')]:
  d=B/folder;rr=read(d/'result.json');f=d/'motion-forecast.pt'
  if sha256(f)!=rr['checkpoint_sha256']:raise ValueError('old checksum')
  old=CausalMotionForecast();old.load_state_dict(torch.load(f,map_location='cpu',weights_only=True)['model']);comparison[name+'_mse']=mse(old,valid)
 with torch.no_grad(): pred,target=arrays(model,valid);hp,_=arrays(hold,valid)
 np.savez_compressed(run/'validation.npz',prediction=pred.numpy(),target=target.numpy(),hold=hp.numpy())
 ck=run/'motion-forecast.pt';torch.save(dict(schema='flybrain.experimental.causal_motion_forecast.v1',model=model.state_dict(),actor_qualified=False,synthetic_causal_training=True),ck)
 out=dict(status='completed_causally_timed_synthetic_forecast_training',optimizer_updates=200,training_loss_first=losses[0],training_loss_last=losses[-1],gains=model.raw_gain.tanh().detach().tolist(),validation=comparison,validation_sha256=h,validation_arrays_sha256=sha256(run/'validation.npz'),checkpoint_sha256=sha256(ck),validation_used_for_fitting=False,only_synthetic_stimulus=True,actor_qualified=False,vehicle_control_sent=False,driving_or_physiology_verified=False)
 write_json(run/'result.json',out);print(json.dumps(out,allow_nan=False))
if __name__=='__main__':main()
