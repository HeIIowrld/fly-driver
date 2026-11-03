"""Inspect contrast reference on fit-forbidden historical AC scenes, without fitting."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.episode_storage import load_array,resolve_payload_path
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
root=B/'20261029-t1434-body21-flight-cache';mp=root/'dataset.json';meta=read(mp);dr=read(root/'result.json');sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json');cp=B/'20261101-corner-preserving-visual-training-1/joint-visual-candidate.pt';cr=read(cp.parent/'result.json')
if sha256(mp)!=dr['dataset_manifest_sha256'] or sha256(sp)!=sr['checkpoint_sha256'] or sha256(cp)!=cr['candidate_sha256']:raise ValueError('exact source integrity required')
cand=torch.load(cp,map_location='cpu',weights_only=True);source=torch.load(sp,map_location='cpu',weights_only=True)
if cand['actor_qualified'] is not False or cand['source_spatial_sha256']!=sha256(sp):raise ValueError('research candidate required')
ref=source['reference'].numpy();lim=float(source['max_contrast_rms']);W=cand['spatial_weight'].numpy();run=begin_run(B/'20261101-historical-reference-range-audit-1',dict(program_sha256=sha256(__file__),dataset_manifest_sha256=sha256(mp),candidate_sha256=sha256(cp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def stats(scene):
 raw=scene.reshape(-1,4,9,16)[:,:,0,:].astype('float64');base=raw-raw.mean(axis=-1,keepdims=True);center=base-ref[None];r0=np.sqrt(np.mean(base**2,axis=(1,2)));r1=np.sqrt(np.mean(center**2,axis=(1,2)));gain=np.minimum(1.,lim/np.maximum(r1,1e-12));pre=(center*gain[:,None,None]).reshape(-1,64)@W.T
 return dict(frames=len(raw),raw_centered_rms_median=float(np.median(r0)),reference_adjusted_rms_median=float(np.median(r1)),adjusted_over_unadjusted_median=float(np.median(r1/np.maximum(r0,1e-12))),gain_median=float(np.median(gain)),gain_min=float(gain.min()),gain_max=float(gain.max()),fraction_clamped=float(np.mean(gain<1)),preactivation_abs_p95=float(np.quantile(np.abs(pre),.95)),fraction_abs_preactivation_over_3=float(np.mean(np.abs(pre)>3)),per_view_unadjusted_rms_median=np.median(np.sqrt(np.mean(base**2,axis=2)),axis=0).tolist())
try:
 episodes=[]
 for name in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):
  e=next(x for x in meta['episodes'] if x['file']==name);file=resolve_payload_path(root/name);record=Path(e['original']['recording']);receipt=read(record/'result.json')
  if sha256(file)!=e['sha256'] or receipt.get('training_admissible') is not False:raise ValueError('verified fit-forbidden historical source required')
  with load_array(root/name,allow_pickle=False) as z:scene=z['scene_features'].copy()
  if scene.shape!=(e['observations'],576) or not np.isfinite(scene).all():raise ValueError('complete recorded scene required')
  episodes.append(dict(episode=name,source_sha256=sha256(file),training_admissible=False,statistics=stats(scene)))
 result.update(status='completed_historical_reference_range_audit',episodes=episodes,reference_rms=float(np.sqrt(np.mean(ref.astype('float64')**2))),contrast_limit=lim,fit_on_real=False,optimizer_updates=0,interpretation='Previously examined fit-forbidden moving AI recordings, cached pooled scenes without reconstructed view-time validity; descriptive contrast range only, not real-time safety, trajectory or physical synchronization')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_historical_reference_range_audit':raise SystemExit(1)
