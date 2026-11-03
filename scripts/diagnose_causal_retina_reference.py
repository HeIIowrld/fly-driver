"""Causal reference diagnostic on archived pooled scenes; never an actor."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.flight_circuit import FlightScenePool
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
pp=B/'20261101-causal-retina-reference-diagnostic-plan/plan.json';plan=read(pp);root=B/'20261029-t1434-body21-flight-cache';mp=root/'dataset.json';meta=read(mp);dr=read(root/'result.json');sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json');cp=B/'20261101-corner-preserving-visual-training-1/joint-visual-candidate.pt';cr=read(cp.parent/'result.json')
if plan['calibration_or_fit_on_real'] is not False or plan['actor_qualified'] is not False or sha256(mp)!=dr['dataset_manifest_sha256'] or sha256(sp)!=sr['checkpoint_sha256'] or sha256(cp)!=cr['candidate_sha256']:raise ValueError('declared fixed sources required')
p=torch.load(sp,map_location='cpu',weights_only=True);c=torch.load(cp,map_location='cpu',weights_only=True)
if c['actor_qualified'] is not False or c['source_spatial_sha256']!=sha256(sp):raise ValueError('fixed candidate mismatch')
ref=p['reference'].numpy().astype('float64');W=c['spatial_weight'].numpy().astype('float64');limit=float(p['max_contrast_rms']);model=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];spec=read(Path(model['spec']))
with load_array(spec['scene_registration_file'],allow_pickle=False) as z:uv=z['image_uv'].copy()
pool=FlightScenePool(uv,spec['scene_basis']).eval();run=begin_run(B/'20261101-causal-retina-reference-diagnostic-1',dict(program_sha256=sha256(__file__),plan_sha256=sha256(pp),candidate_sha256=sha256(cp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def evaluate(scene,t,tau):
 if scene.ndim!=2 or scene.shape[1]!=576 or len(t)!=len(scene) or not np.isfinite(scene).all() or np.any(np.diff(t)<=0):raise ValueError('complete ordered scenes required')
 x=scene.reshape(-1,4,9,16)[:,:,0,:].astype('float64');x=x-x.mean(axis=-1,keepdims=True);res=np.empty_like(x)
 if tau is None:res=x-ref
 else:
  state=x[0].copy();res[0]=0.
  for i in range(1,len(x)):
   dt=float(t[i]-t[i-1]);res[i]=x[i]-state;state=state+(1.-np.exp(-dt/tau))*res[i]
 r=np.sqrt(np.mean(res**2,axis=(1,2)));gain=np.minimum(1.,limit/np.maximum(r,1e-12));pre=(res*gain[:,None,None]).reshape(-1,64)@W.T;out=.5*np.tanh(pre)
 return r,gain,out
try:
 reports=[];arrays={}
 for name in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):
  e=next(x for x in meta['episodes'] if x['file']==name);f=resolve_payload_path(root/name);receipt=read(Path(e['original']['recording'])/'result.json')
  if sha256(f)!=e['sha256'] or receipt.get('training_admissible') is not False:raise ValueError('fit-forbidden history required')
  with load_array(root/name,allow_pickle=False) as z:scene=z['scene_features'].copy();t=z['observation_time'].copy()
  if len(scene)!=e['observations']:raise ValueError('history length mismatch')
  mask=t-t[0]>=1.;modes={}
  for mode,tau in plan['modes'].items():
   r,g,o=evaluate(scene,t,tau);modes[mode]=dict(rms_median=float(np.median(r[mask])),gain_median=float(np.median(g[mask])),fraction_clamped=float(np.mean(g[mask]<1.)),output_abs_mean=float(np.mean(np.abs(o[mask]))));arrays[name.replace('.npz','')+'_'+mode+'_gain']=g
  reports.append(dict(episode=name,source_sha256=sha256(f),frames=len(scene),analysis_after_seconds=1.,modes=modes))
 synthetic=[]
 for seed in (881605,881606):
  folder=B/f'20261101-corner-preserving-validation-{seed}';rr=read(folder/'result.json');f=folder/'responses.npz'
  if sha256(f)!=rr['responses_sha256']:raise ValueError('synthetic source mismatch')
  with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();t=z['times'].copy();bends=z['bend'].copy();yaws=z['yaw_rate'].copy()
  if not np.array_equal(bends,[0,-1,1]) or not np.allclose(yaws,[0,-.12,.12]) or ret.shape!=(3,3,41,4,721):raise ValueError('synthetic grid mismatch')
  x=np.zeros((369,4,9,721),dtype='float32');x[:,:,0]=ret.reshape(369,4,721)
  with torch.no_grad():sc=pool(torch.from_numpy(x)).numpy().reshape(3,3,41,576)
  modes={}
  for mode,tau in plan['modes'].items():
   out=np.stack([[evaluate(sc[b,j],t,tau)[2] for j in range(3)] for b in range(3)]);yd=out[0,2,-10:]-out[0,1,-10:];cd=out[1,0,-10:]-out[2,0,-10:];modes[mode]=dict(yaw_port_difference_rms=float(np.sqrt(np.mean(yd**2))),corner_port_difference_rms=float(np.sqrt(np.mean(cd**2))));arrays[f'{seed}_{mode}_ports']=out
  synthetic.append(dict(seed=seed,source_sha256=sha256(f),modes=modes))
 out=run/'diagnostic.npz';np.savez_compressed(out,**arrays);result.update(status='completed_causal_retina_reference_diagnostic',historical=reports,synthetic=synthetic,plan_sha256=sha256(pp),output_sha256=sha256(out),fit_on_real=False,optimizer_updates=0,actor_qualified=False,interpretation='Experimental reference state on previously inspected fit-forbidden pooled scenes, no reconstructed per-view freshness; boundary output only, not measured fly adaptation, circuit causality or driving')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_causal_retina_reference_diagnostic':raise SystemExit(1)
