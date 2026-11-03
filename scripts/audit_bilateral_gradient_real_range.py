"""Evaluate gradient-direction counterfactual range on fit-forbidden real caches."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.bilateral_retinal_contrast_boundary import BilateralRetinalContrastBoundary
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'));root=B/'20261029-t1434-body21-flight-cache';mp=root/'dataset.json';meta=read(mp);dr=read(root/'result.json');ck=B/'20261101-bilateral-spatial-training-1/bilateral-spatial-candidate.pt';cr=read(ck.parent/'result.json');gd=B/'20261101-bilateral-gradient-direction-1';gr=read(gd/'result.json');df=gd/'direction.npz'
if sha256(mp)!=dr['dataset_manifest_sha256'] or sha256(ck)!=cr['candidate_sha256'] or gr['status']!='completed_synthetic_bilateral_gradient_direction' or sha256(df)!=gr['direction_sha256']:raise ValueError('verified inputs required')
c=torch.load(ck,map_location='cpu',weights_only=True)
if c['actor_qualified'] is not False:raise ValueError('research only')
model=BilateralRetinalContrastBoundary(c['spatial_model']['projection.weight'].shape[0],normalization_floor=c['normalization_floor']).eval();model.load_state_dict(c['spatial_model'],strict=True)
with np.load(df,allow_pickle=False) as z:d=z['direction'].copy()
if d.shape!=tuple(model.projection.weight.shape) or not np.isclose(np.linalg.norm(d),1.,atol=1e-5):raise ValueError('unit direction required')
W=model.projection.weight.detach().numpy().astype('float64');run=begin_run(B/'20261101-bilateral-gradient-real-range-1',dict(program_sha256=sha256(__file__),candidate_sha256=sha256(ck),direction_sha256=sha256(df),dataset_sha256=sha256(mp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
try:
 rows=[]
 for name in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):
  e=next(x for x in meta['episodes'] if x['file']==name);f=resolve_payload_path(root/name);rec=read(Path(e['original']['recording'])/'result.json')
  if sha256(f)!=e['sha256'] or rec.get('training_admissible') is not False:raise ValueError('fit-forbidden source required')
  with load_array(root/name,allow_pickle=False) as z:scene=z['scene_features'].copy()
  x=torch.tensor(scene,dtype=torch.float32);valid=torch.ones((len(x),4),dtype=torch.bool)
  with torch.no_grad():features=model.features(x,valid).numpy().astype('float64')
  modes={}
  for step in (0.,1.,5.,20.):
   pre=features@(W+step*d).T;deriv=.5*(1.-np.tanh(pre)**2);modes[str(step)]=dict(pre_abs_p95=float(np.quantile(np.abs(pre),.95)),pre_abs_gt3_fraction=float(np.mean(np.abs(pre)>3)),derivative_lt_001_fraction=float(np.mean(deriv<.01)))
  rows.append(dict(episode=name,source_sha256=sha256(f),frames=len(x),modes=modes))
 result.update(status='completed_bilateral_gradient_real_range',episodes=rows,fit_on_real=False,all_views_assumed_valid_for_offline_range=True,optimizer_updates=0,interpretation='Post-hoc fixed synthetic gradient perturbations on known fit-forbidden scene caches; no per-view timing validity or actual vehicle control')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_bilateral_gradient_real_range':raise SystemExit(1)
