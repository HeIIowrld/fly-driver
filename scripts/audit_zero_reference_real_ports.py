"""Fixed-vs-zero contrast reference on previously reviewed, fit-forbidden history."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.episode_storage import load_array,resolve_payload_path
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
planpath=B/'20261101-zero-retinal-reference-plan/plan.json';plan=read(planpath);root=B/'20261029-t1434-body21-flight-cache';mp=root/'dataset.json';meta=read(mp);dr=read(root/'result.json');sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json');cp=B/plan['candidate'];cr=read(cp.parent/'result.json')
if plan['fit_on_real'] is not False or plan['actor_qualified'] is not False or sha256(mp)!=dr['dataset_manifest_sha256'] or sha256(sp)!=sr['checkpoint_sha256'] or sha256(cp)!=cr['candidate_sha256']:raise ValueError('verified fixed sources required')
c=torch.load(cp,map_location='cpu',weights_only=True);p=torch.load(sp,map_location='cpu',weights_only=True)
if c['actor_qualified'] is not False or c['source_spatial_sha256']!=sha256(sp):raise ValueError('candidate lineage')
ref=p['reference'].numpy();limit=float(p['max_contrast_rms']);W=c['spatial_weight'].numpy();run=begin_run(B/'20261101-zero-reference-real-port-audit-1',dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),candidate_sha256=sha256(cp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
try:
 rows=[]
 for name in plan['actual_episodes']:
  e=next(x for x in meta['episodes'] if x['file']==name);f=resolve_payload_path(root/name);rec=read(Path(e['original']['recording'])/'result.json')
  if sha256(f)!=e['sha256'] or rec.get('training_admissible') is not False:raise ValueError('fit-forbidden source required')
  with load_array(root/name,allow_pickle=False) as z:x=z['scene_features'].copy().reshape(-1,4,9,16)[:,:,0,:].astype('float64')
  x=x-x.mean(axis=-1,keepdims=True);outputs={};details={}
  for mode,offset in [('fixed_synthetic_reference',ref),('zero_reference',np.zeros_like(ref))]:
   res=x-offset;raw=np.sqrt(np.mean(res**2,axis=(1,2)));gain=np.minimum(1,limit/np.maximum(raw,1e-12));pre=(res*gain[:,None,None]).reshape(-1,64)@W.T;out=.5*np.tanh(pre);outputs[mode]=out;details[mode]=dict(gain_median=float(np.median(gain)),fraction_clamped=float(np.mean(gain<1)),pre_abs_p95=float(np.quantile(np.abs(pre),.95)),output_abs_mean=float(np.mean(np.abs(out))))
  diff=outputs['zero_reference']-outputs['fixed_synthetic_reference'];rows.append(dict(episode=name,frames=len(x),source_sha256=sha256(f),modes=details,mean_abs_port_difference=float(np.mean(np.abs(diff))),max_abs_port_difference=float(np.max(np.abs(diff)))))
 result.update(status='completed_zero_reference_real_port_audit',episodes=rows,plan_sha256=sha256(planpath),fit_on_real=False,optimizer_updates=0,interpretation='Read-only zero-reference intervention on known fit-forbidden cached moving scenes; per-view freshness absent, output is bounded engineering port not actor or vehicle control')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_zero_reference_real_port_audit':raise SystemExit(1)
