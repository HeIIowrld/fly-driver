"""Compare synthetic yaw-opponent visual tensors across two textures."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
b=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');up=b/'20261101-synthetic-matched-body-yaw-1';ur=json.loads((up/'result.json').read_text(encoding='utf8'));data=up/'yaw-body.npz'
if ur['status']!='completed_synthetic_matched_body_yaw' or sha256(data)!=ur['output_sha256']:raise ValueError('prior run mismatch')
roots=[b/'20261101-synthetic-road-yaw-881403',b/'20261101-multispeed-eval-881511-3'];vals=[];sources=[]
for seed,root in zip((881403,881511),roots):
 r=json.loads((root/'result.json').read_text(encoding='utf8'));cfg=json.loads((root/'manifest.json').read_text(encoding='utf8'))['config'];p=root/'responses.npz'
 if cfg['texture_seed']!=seed or r['responses_sha256']!=sha256(p) or r['status']!='completed_causal_synthetic_road_with_yaw':raise ValueError('source mismatch')
 with np.load(p,allow_pickle=False) as z:
  if not np.allclose(z['yaw_rate'],[0,-.12,.12]) or z['bend'][0]!=0:raise ValueError('yaw coordinates')
  vals.append({k:(z[k][0,2,-10:].astype('float64')-z[k][0,1,-10:].astype('float64')) for k in ('retina','motion')})
 sources.append(sha256(p))
run=begin_run(b/'20261101-yaw-texture-visual-correlation-1',dict(program_sha256=sha256(__file__),source_sha256=sources,upstream_sha256=sha256(data),optimizer_updates=0));out=dict(status='failed',vehicle_control_sent=False,actor_qualified=False)
def metrics(x,y):
 a=x.ravel();c=y.ravel();na=np.linalg.norm(a);nc=np.linalg.norm(c)
 return dict(cosine=float(np.dot(a,c)/(na*nc)) if na and nc else None,rms_a=float(np.sqrt(np.mean(a*a))),rms_b=float(np.sqrt(np.mean(c*c))),same_sign_fraction=float(np.mean(np.sign(a)==np.sign(c))))
try:
 results={}
 for k in ('retina','motion'):
  a=vals[0][k];c=vals[1][k];results[k]=dict(overall=metrics(a,c),per_view={name:metrics(a[:,i],c[:,i]) for i,name in enumerate(('front','left','right','rear'))})
  if k=='motion':results[k]['t4']=metrics(a[:,:,0:4],c[:,:,0:4]);results[k]['t5']=metrics(a[:,:,4:8],c[:,:,4:8])
 out.update(status='completed_yaw_texture_visual_correlation',visual=results,source_sha256=sources,prior_result_sha256=sha256(up/'result.json'),interpretation='Same pixels/neural columns and correlated late frames for +yaw minus -yaw; cosine between different textures is descriptive, not decoding, causality or measured physiology')
except Exception as e:out.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',out)
print(json.dumps(out,allow_nan=False))
if out['status']!='completed_yaw_texture_visual_correlation':raise SystemExit(1)
