"""Trace pooled retinal yaw differences through the frozen spatial boundary."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
rroot=B/'20261101-corner-preserving-visual-training-1';rr=read(rroot/'result.json');cp=rroot/'joint-visual-candidate.pt';sroot=B/'20261101-adaptive-pattern-fit';sr=read(sroot/'result.json');sp=sroot/'adaptive-pattern-contrast.pt'
if rr['status']!='completed_synthetic_corner_preserving_visual_training' or sha256(cp)!=rr['candidate_sha256'] or sha256(sp)!=sr['checkpoint_sha256']:raise ValueError('source mismatch')
cand=torch.load(cp,map_location='cpu',weights_only=True);orig=torch.load(sp,map_location='cpu',weights_only=True)
if cand['actor_qualified'] is not False or cand['source_spatial_sha256']!=sha256(sp):raise ValueError('candidate mismatch')
mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];spec=read(Path(mp['spec']))
with load_array(spec['scene_registration_file'],allow_pickle=False) as z:uv=z['image_uv'].copy()
pool=FlightScenePool(uv,spec['scene_basis']).eval();boundary=RetinalSpatialContrastBoundary(cand['spatial_weight'].shape[0],reference=orig['reference'],max_contrast_rms=orig['max_contrast_rms']).eval()
with torch.no_grad():boundary.projection.weight.copy_(cand['spatial_weight'])
run=begin_run(B/'20261101-spatial-yaw-normalization-audit-1',dict(program_sha256=sha256(__file__),candidate_sha256=sha256(cp),spatial_source_sha256=sha256(sp),optimizer_updates=0));out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def rms(x):return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
def cosine(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else None
try:
 reports=[];details={};saved={}
 for seed in (881605,881606):
  root=B/f'20261101-corner-preserving-validation-{seed}';r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if cfg['texture_seed']!=seed or r['responses_sha256']!=sha256(f) or r['status']!='completed_causal_synthetic_road_with_yaw':raise ValueError('stimulus lineage')
  with np.load(f,allow_pickle=False) as z:
   if not np.allclose(z['yaw_rate'],[0,-.12,.12]) or z['bend'][0]!=0:raise ValueError('declared straight yaw')
   retina=z['retina'][0,:,-10:].copy()
  x=np.zeros((30,4,9,721),dtype='float32');x[:,:,0]=retina.reshape(30,4,721)
  with torch.no_grad():
   scene=pool(torch.from_numpy(x));raw=scene.reshape(-1,4,9,16)[:,:,0,:];centered=raw-raw.mean(dim=-1,keepdim=True)-boundary.reference;rms_tensor=(centered.square().mean(dim=(1,2),keepdim=True)+1e-24).sqrt();gain=(boundary.max_contrast_rms/rms_tensor).clamp(max=1.);normalized=centered*gain;pre=boundary.projection(normalized.reshape(-1,64));output=boundary(scene,torch.ones((30,4),dtype=torch.bool))
  feature=normalized.numpy().reshape(3,10,4,16);pre=pre.numpy().reshape(3,10,-1);output=output.numpy().reshape(3,10,-1);dg=feature[2]-feature[1];dp=pre[2]-pre[1];do=output[2]-output[1];w=boundary.projection.weight.detach().numpy().reshape(-1,4,16);view_contrib=np.einsum('tvd,pvd->tvp',dg,w)
  details[seed]=dict(feature_delta=dg,pre_delta=dp,out_delta=do,view_contrib=view_contrib);saved[f'{seed}_feature_delta']=dg;saved[f'{seed}_pre_delta']=dp;saved[f'{seed}_view_contrib']=view_contrib
  reports.append(dict(seed=seed,source_sha256=sha256(f),gain_min=float(gain.min()),gain_max=float(gain.max()),fraction_clamped=float((gain<1.).float().mean()),centered_rms_min=float(rms_tensor.min()),centered_rms_max=float(rms_tensor.max()),yaw_feature_rms_by_view=[rms(dg[:,v]) for v in range(4)],yaw_preprojection_rms_by_view=[rms(view_contrib[:,v]) for v in range(4)],yaw_preprojection_total_rms=rms(dp),yaw_output_total_rms=rms(do)))
 a,c=details[881605],details[881606];cross=dict(feature_cosine=cosine(a['feature_delta'],c['feature_delta']),pre_projection_cosine=cosine(a['pre_delta'],c['pre_delta']),output_cosine=cosine(a['out_delta'],c['out_delta']),per_view_input_cosine=[cosine(a['feature_delta'][:,v],c['feature_delta'][:,v]) for v in range(4)],per_view_preprojection_cosine=[cosine(a['view_contrib'][:,v],c['view_contrib'][:,v]) for v in range(4)])
 path=run/'spatial-yaw-vectors.npz';np.savez_compressed(path,**saved);out.update(status='completed_spatial_yaw_normalization_audit',conditions=reports,cross_texture=cross,output_sha256=sha256(path),interpretation='Fixed-pooling frozen engineering spatial projection, +/-yaw differences on late correlated synthetic frames; projection cosine is not steering causality or physiological validation')
except Exception as e:out.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',out)
print(json.dumps(out,allow_nan=False))
if out['status']!='completed_spatial_yaw_normalization_audit':raise SystemExit(1)
