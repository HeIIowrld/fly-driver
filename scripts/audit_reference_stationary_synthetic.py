"""Inspect fixed synthetic contrast reference without fitting or actor use."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_circuit import FlightScenePool
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json');vr=B/'20261101-stationary-native-flyvis-geometry-replay-v2';vf=vr/'responses.npz';rr=read(vr/'result.json');capture=B/'20261101-stationary-native-only-fourview-capture';cap=read(capture/'result.json')
if sha256(sp)!=sr['checkpoint_sha256'] or sha256(vf)!=rr['responses_sha256'] or cap['training_admissible'] is not False:raise ValueError('fixed sources required')
p=torch.load(sp,map_location='cpu',weights_only=True);ref=p['reference'].numpy().astype('float64');limit=float(p['max_contrast_rms']);mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];spec=read(Path(mp['spec']))
with load_array(spec['scene_registration_file'],allow_pickle=False) as z:uv=z['image_uv'].copy()
pool=FlightScenePool(uv,spec['scene_basis']).eval();run=begin_run(B/'20261101-reference-stationary-synthetic-audit-1',dict(program_sha256=sha256(__file__),source_spatial_sha256=sha256(sp),actual_source_sha256=sha256(vf),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def stats(retina):
 x=np.zeros((len(retina),4,9,721),dtype='float32');x[:,:,0]=retina
 with torch.no_grad():raw=pool(torch.from_numpy(x)).numpy().reshape(-1,4,9,16)[:,:,0].astype('float64')
 base=raw-raw.mean(axis=-1,keepdims=True);after=base-ref[None];r0=np.sqrt(np.mean(base**2,axis=(1,2)));r1=np.sqrt(np.mean(after**2,axis=(1,2)));g=np.minimum(1,limit/np.maximum(r1,1e-12));return dict(frames=len(retina),raw_centered_rms_median=float(np.median(r0)),reference_adjusted_rms_median=float(np.median(r1)),adjusted_over_unadjusted_median=float(np.median(r1/np.maximum(r0,1e-12))),gain_median=float(np.median(g)),fraction_clamped=float(np.mean(g<1.)))
try:
 events=[json.loads(x) for x in (capture/'client/events.jsonl').read_text(encoding='utf8').splitlines()];valid=np.array([all(float(v)>0 for v in e['summary']['view_valid_counts']) for e in events])
 with np.load(vf,allow_pickle=False) as z:ret=z['retina'].copy()
 if len(ret)!=24 or valid.sum()!=23:raise ValueError('actual validity mismatch')
 results={'actual_stationary':stats(ret[valid])};hashes={}
 for seed in (881605,881606):
  root=B/f'20261101-corner-preserving-validation-{seed}';r=read(root/'result.json');f=root/'responses.npz'
  if sha256(f)!=r['responses_sha256']:raise ValueError('synthetic evidence mismatch')
  with np.load(f,allow_pickle=False) as z:ret=z['retina'][0,:,-10:].reshape(30,4,721).copy()
  results[str(seed)]=stats(ret);hashes[str(seed)]=sha256(f)
 result.update(status='completed_reference_stationary_synthetic_audit',conditions=results,reference_rms=float(np.sqrt(np.mean(ref**2))),contrast_limit=limit,synthetic_source_sha256=hashes,optimizer_updates=0,fit_on_real=False,interpretation='Previously known AC stationary and synthetic visual data, contrast reference counterfactual only; no fitted real data, matched dynamics, physical exposure or driving')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_reference_stationary_synthetic_audit':raise SystemExit(1)
