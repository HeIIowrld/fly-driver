"""Offline range audit of bilateral boundary on fit-forbidden real scene caches."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.bilateral_retinal_contrast_boundary import BilateralRetinalContrastBoundary
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'));root=B/'20261029-t1434-body21-flight-cache';mp=root/'dataset.json';m=read(mp);dr=read(root/'result.json');ck=B/'20261101-bilateral-spatial-training-1/bilateral-spatial-candidate.pt';rr=read(ck.parent/'result.json')
if sha256(mp)!=dr['dataset_manifest_sha256'] or sha256(ck)!=rr['candidate_sha256']:raise ValueError('exact recorded sources required')
c=torch.load(ck,map_location='cpu',weights_only=True)
if c['actor_qualified'] is not False:raise ValueError('research candidate only')
model=BilateralRetinalContrastBoundary(c['spatial_model']['projection.weight'].shape[0],normalization_floor=c['normalization_floor']).eval();model.load_state_dict(c['spatial_model'],strict=True)
run=begin_run(B/'20261101-bilateral-spatial-real-range-1',dict(program_sha256=sha256(__file__),candidate_sha256=sha256(ck),dataset_sha256=sha256(mp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
try:
 rows=[]
 for name in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):
  e=next(x for x in m['episodes'] if x['file']==name);file=resolve_payload_path(root/name);r=read(Path(e['original']['recording'])/'result.json')
  if sha256(file)!=e['sha256'] or r.get('training_admissible') is not False:raise ValueError('fit forbidden source required')
  with load_array(root/name,allow_pickle=False) as z:scene=z['scene_features'].copy()
  x=torch.tensor(scene,dtype=torch.float32);valid=torch.ones((len(x),4),dtype=torch.bool)
  with torch.no_grad():features=model.features(x,valid);pre=model.projection(features);out=model(x,valid);deriv=.5*(1-torch.tanh(pre)**2)
  rows.append(dict(episode=name,frames=len(x),source_sha256=sha256(file),feature_abs_p95=float(torch.quantile(features.abs(),.95)),pre_abs_p95=float(torch.quantile(pre.abs(),.95)),saturated_abs_pre_over3=float((pre.abs()>3).float().mean()),derivative_lt_001=float((deriv<.01).float().mean()),output_abs_mean=float(out.abs().mean())))
 result.update(status='completed_bilateral_spatial_real_range',episodes=rows,fit_on_real=False,optimizer_updates=0,all_views_assumed_valid_for_offline_range=True,interpretation='Previously examined fit-forbidden moving scene cache; assumed validity is not real-time admission, and port range is not driving, physiology, or correctness')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_bilateral_spatial_real_range':raise SystemExit(1)
