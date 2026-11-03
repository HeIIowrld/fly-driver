"""Counterfactual contrast cap on fit-forbidden cached real camera scenes."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.episode_storage import load_array,resolve_payload_path
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
root=B/'20261029-t1434-body21-flight-cache';mp=root/'dataset.json';meta=read(mp);dr=read(root/'result.json');sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json');cp=B/'20261101-corner-preserving-visual-training-1/joint-visual-candidate.pt';cr=read(cp.parent/'result.json')
if sha256(mp)!=dr['dataset_manifest_sha256'] or sha256(sp)!=sr['checkpoint_sha256'] or sha256(cp)!=cr['candidate_sha256']:raise ValueError('verified data lineage required')
p=torch.load(sp,map_location='cpu',weights_only=True);c=torch.load(cp,map_location='cpu',weights_only=True)
if c['actor_qualified'] is not False or c['source_spatial_sha256']!=sha256(sp):raise ValueError('diagnostic candidate only')
W=c['spatial_weight'].numpy().astype('float64');limit=float(p['max_contrast_rms']);run=begin_run(B/'20261101-real-zero-ref-limit-audit-1',dict(program_sha256=sha256(__file__),candidate_sha256=sha256(cp),dataset_sha256=sha256(mp),cap_multipliers=[1,10,30],optimizer_updates=0));out=dict(status='failed',vehicle_control_sent=False,actor_qualified=False)
try:
 rows=[]
 for name in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):
  e=next(x for x in meta['episodes'] if x['file']==name);f=resolve_payload_path(root/name);r=read(Path(e['original']['recording'])/'result.json')
  if sha256(f)!=e['sha256'] or r.get('training_admissible') is not False:raise ValueError('fit-forbidden source required')
  with load_array(root/name,allow_pickle=False) as z:x=z['scene_features'].copy().reshape(-1,4,9,16)[:,:,0].astype('float64')
  centered=x-x.mean(axis=-1,keepdims=True);rms=np.sqrt(np.mean(centered**2,axis=(1,2)));modes={}
  for scale in (1,10,30):
   gain=np.minimum(1.,limit*scale/np.maximum(rms,1e-12));pre=(centered*gain[:,None,None]).reshape(-1,64)@W.T;derivative=.5*(1.-np.tanh(pre)**2)
   modes[str(scale)]=dict(gain_median=float(np.median(gain)),clamped_fraction=float(np.mean(gain<1)),pre_abs_p95=float(np.quantile(np.abs(pre),.95)),saturated_abs_pre_over3=float(np.mean(np.abs(pre)>3)),tanh_derivative_mean=float(np.mean(derivative)),tanh_derivative_lt_001_fraction=float(np.mean(derivative<.01)))
  rows.append(dict(episode=name,source_sha256=sha256(f),frames=len(x),modes=modes))
 out.update(status='completed_real_zero_ref_limit_audit',episodes=rows,fit_on_real=False,optimizer_updates=0,interpretation='Known fit-forbidden cached images, zero reference and hypothetical cap multipliers; no per-view freshness reconstruction, no biological or vehicle-control claim')
except Exception as e:out.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',out)
print(json.dumps(out,allow_nan=False))
if out['status']!='completed_real_zero_ref_limit_audit':raise SystemExit(1)
