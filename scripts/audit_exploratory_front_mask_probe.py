"""Exploratory equal-area front-view masking in stationary real and synthetic pixels."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.graph import sha256
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();live=B/'20261101-stationary-unified-held-prefix';lr=read(live/'result.json');events=read(live/'events.json');archive=live/'vision';bundle=read(archive/'bundle.json');manifest=read(archive/'archive.json');zp=archive/'camera.zip';probe=B/'20261101-road-signal-decodability';pr=read(probe/'result.json');wf=probe/'diagnostic.npz';poolfile=B/'20261101-road-scene-pool-audit/pool-audit.npz';sources=B/'20261101-joint-vision-yaw-new-sources';plan=read(B/'20261101-joint-vision-yaw-new-plan/plan.json')
 if sha256(live/'events.json')!=lr['events_sha256'] or sha256(zp)!=bundle['sha256'] or sha256(archive/'archive.json')!=bundle['archive_manifest_sha256'] or sha256(wf)!=pr['outputs_sha256']:raise ValueError('Exact original evidence required')
 with np.load(wf,allow_pickle=False) as z:w=z['pooled_weights'].copy().reshape(4,16);center=z['pooled_center'].copy().reshape(4,16)
 with np.load(poolfile,allow_pickle=False) as z:pool=z['pool_weight'].copy()
 camera=read('configs/ac-csp-exterior-front-v3-retina.json');uv=[(q,r) for q in range(-15,16) for r in range(max(-15,-15-q),min(15,15-q)+1)];projector=FlyvisCameraProjector(camera,*np.array(uv).T)
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),archive_sha256=sha256(zp),probe_sha256=sha256(wf),scope='exploratory post-observation equal-area front masking; no fitting',optimizer_updates=0,actor=False));result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False}
 try:
  def features(images):return np.tanh(np.stack([projector.sample(x) for x in images]).astype(np.float64))@pool.T
  def scores(images):
   original=[x.copy() for x in images];bottom=[x.copy() for x in images];top=[x.copy() for x in images];bottom[0][117:,:,:]=128;top[0][:63,:,:]=128
   return np.array([((features(x)-center)*w).sum() for x in (original,bottom,top)])
  expected={x['file']:x for x in manifest};real=[]
  with zipfile.ZipFile(zp) as z:
   for event in events:
    group=[]
    for name in ('front','left','right','rear'):
     file=f"{event['capture_id']:04d}-{name}.png";raw=z.read(file)
     if hashlib.sha256(raw).hexdigest()!=expected[file]['sha256']:raise ValueError('Original image hash mismatch')
     group.append(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy())
    real.append(scores(group))
  synthetic=[];source_hashes=[]
  for entry in plan['validation']:
   root=sources/f"validation-{entry['seed']}";r=read(root/'result.json');f=root/'responses.npz'
   if sha256(f)!=r['responses_sha256'] or sha256('src/flybrain_drive/synthetic_road_visual.py')!=plan['road_source_sha256']:raise ValueError('Original synthetic source mismatch')
   road=SyntheticRoadVisual(camera,curvature=entry['curvature'],bend_start_m=entry['bend_start_m'])
   with np.load(f,allow_pickle=False) as z:stored=z['retina'].copy()
   for index,bend in enumerate((0,-1,1)):
    images=road.render(bend=bend,forward_m=3.6,texture_seed=entry['seed']);sample=np.stack([projector.sample(x) for x in images])
    if not np.allclose(sample,stored[index,-1],atol=1e-6,rtol=0):raise ValueError('Synthetic pixel recreation mismatch')
    synthetic.append(scores(images))
   source_hashes.append(sha256(f))
  real=np.stack(real);synthetic=np.stack(synthetic);out=run/'mask-scores.npz';np.savez_compressed(out,real=real,synthetic=synthetic)
  result.update(status='completed_exploratory_front_mask_probe',real_groups=len(real),synthetic_images=len(synthetic),modes=['original','front_bottom_35pct_gray','front_top_35pct_gray'],real_score_mean=real.mean(axis=0).tolist(),synthetic_score_mean=synthetic.mean(axis=0).tolist(),mean_domain_difference=(real.mean(axis=0)-synthetic.mean(axis=0)).tolist(),real_mask_score_change=(real[:,1:]-real[:,:1]).mean(axis=0).tolist(),synthetic_mask_score_change=(synthetic[:,1:]-synthetic[:,:1]).mean(axis=0).tolist(),output_sha256=sha256(out),fit_on_real=False,interpretation='Exploratory post-inspection masks replace image pixels with artificial gray, only for a synthetic diagnostic probe; no trained model or actor. Stationary frame and synthetic generation differ beyond cockpit, and masks do not establish a specific physical cause.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_exploratory_front_mask_probe':raise SystemExit(1)
if __name__=='__main__':main()
