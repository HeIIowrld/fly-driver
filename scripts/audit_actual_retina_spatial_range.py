"""Compare spatial boundary operating range on archived AC and synthetic retina."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
vr=B/'20261101-stationary-native-flyvis-geometry-replay-v2';vf=vr/'responses.npz';v=read(vr/'result.json');capture=B/'20261101-stationary-native-only-fourview-capture';cap=read(capture/'result.json');sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json');cand=B/'20261101-corner-preserving-visual-training-1/joint-visual-candidate.pt';cr=read(cand.parent/'result.json')
if v['status']!='completed_offline_visual_probe' or sha256(vf)!=v['responses_sha256'] or cap['status']!='completed' or cap['training_admissible'] is not False or sha256(sp)!=sr['checkpoint_sha256'] or sha256(cand)!=cr['candidate_sha256']:raise ValueError('verified archived source required')
orig=torch.load(sp,map_location='cpu',weights_only=True);cp=torch.load(cand,map_location='cpu',weights_only=True)
if cp['actor_qualified'] is not False or cp['source_spatial_sha256']!=sha256(sp):raise ValueError('unqualified research candidate required')
mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];spec=read(Path(mp['spec']))
with load_array(spec['scene_registration_file'],allow_pickle=False) as z:uv=z['image_uv'].copy()
pool=FlightScenePool(uv,spec['scene_basis']).eval();bound=RetinalSpatialContrastBoundary(cp['spatial_weight'].shape[0],reference=orig['reference'],max_contrast_rms=orig['max_contrast_rms']).eval()
with torch.no_grad():bound.projection.weight.copy_(cp['spatial_weight'])
run=begin_run(B/'20261101-actual-retina-spatial-range-audit-1',dict(program_sha256=sha256(__file__),actual_responses_sha256=sha256(vf),capture_result_sha256=sha256(capture/'result.json'),candidate_sha256=sha256(cand),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def stats(retina):
 x=np.zeros((len(retina),4,9,721),dtype='float32');x[:,:,0]=retina
 with torch.no_grad():
  sc=pool(torch.from_numpy(x));raw=sc.reshape(-1,4,9,16)[:,:,0];center=raw-raw.mean(dim=-1,keepdim=True)-bound.reference;rms=(center.square().mean(dim=(1,2),keepdim=True)+1e-24).sqrt();gain=(bound.max_contrast_rms/rms).clamp(max=1.);pre=bound.projection((center*gain).reshape(-1,64));out=bound(sc,torch.ones((len(x),4),dtype=torch.bool))
 values=rms.numpy().ravel();g=gain.numpy().ravel();p=pre.numpy();o=out.numpy()
 return dict(frames=len(x),centered_rms_min=float(values.min()),centered_rms_median=float(np.median(values)),centered_rms_max=float(values.max()),gain_min=float(g.min()),gain_median=float(np.median(g)),gain_max=float(g.max()),fraction_clamped=float(np.mean(g<1.)),preactivation_abs_median=float(np.median(np.abs(p))),preactivation_abs_p95=float(np.quantile(np.abs(p),.95)),fraction_abs_preactivation_over_3=float(np.mean(np.abs(p)>3.)),output_abs_mean=float(np.mean(np.abs(o))),mean_abs_step=float(np.mean(np.abs(np.diff(o,axis=0)))) if len(x)>1 else None)
try:
 events=[json.loads(x) for x in (capture/'client/events.jsonl').read_text(encoding='utf8').splitlines()];valid=np.asarray([all(float(v)>0 for v in e['summary']['view_valid_counts']) for e in events]);
 with np.load(vf,allow_pickle=False) as z:actual=z['retina'].copy()
 if actual.shape!=(24,4,721) or len(valid)!=24 or valid.sum()!=23:raise ValueError('captured frame/validity mismatch')
 conditions={'actual_archived_valid':stats(actual[valid])}
 for seed in (881605,881606):
  root=B/f'20261101-corner-preserving-validation-{seed}';rr=read(root/'result.json');f=root/'responses.npz'
  if sha256(f)!=rr['responses_sha256']:raise ValueError('synthetic hash mismatch')
  with np.load(f,allow_pickle=False) as z:ret=z['retina'][0,:,-10:].reshape(30,4,721).copy()
  conditions[f'synthetic_{seed}']=stats(ret)
 result.update(status='completed_actual_retina_spatial_range_audit',conditions=conditions,actual_source_sha256=sha256(vf),actual_capture_result_sha256=sha256(capture/'result.json'),camera_geometry_actual_sha256=sha256('configs/ac-csp-yaw-retina.json'),camera_geometry_synthetic_sha256=sha256('configs/ac-csp-exterior-front-v3-retina.json'),physical_exposure_verified=False,training_allowed=False,interpretation='Offline single stationary AC archive versus synthetic yaw stimuli; different camera calibrations, source motion, timing and frames; range comparison is not matched distribution, driving or physiology')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_actual_retina_spatial_range_audit':raise SystemExit(1)
