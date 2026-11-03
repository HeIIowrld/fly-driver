"""Compare a frozen synthetic retinal boundary on synthetic and fit-forbidden historical inputs."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.graph import sha256
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();fit=B/'20261101-pattern-port-fit';fr=read(fit/'result.json');ck=fit/'pattern-contrast.pt'
 if fr['status']!='completed_synthetic_pattern_port_fit' or sha256(ck)!=fr['checkpoint_sha256']:raise ValueError('candidate mismatch')
 src=B/'20261029-t1434-body21-flight-cache';dr=read(src/'result.json');manifest=src/'dataset.json'
 if sha256(manifest)!=dr['dataset_manifest_sha256']:raise ValueError('dataset mismatch')
 ep=next(e for e in read(manifest)['episodes'] if e['file']=='episode-06.npz');record=Path(ep['original']['recording']);receipt=read(record/'result.json')
 if receipt.get('training_admissible') is not False or sha256(resolve_payload_path(src/ep['file']))!=ep['sha256']:raise ValueError('fit-forbidden source required')
 audit=B/'20261101-road-scene-pool-audit';ar=read(audit/'result.json');af=audit/'pool-audit.npz'
 if sha256(af)!=ar['outputs_sha256']:raise ValueError('pool provenance')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(ck),dataset_sha256=sha256(manifest),fit_forbidden_historical=True,actor=False))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  torch.set_num_threads(1);payload=torch.load(ck,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('unqualified checkpoint required')
  boundary=RetinalSpatialContrastBoundary(len(payload['weight']),reference=payload['reference']);boundary.projection.weight.data.copy_(payload['weight']);boundary.eval()
  with load_array(src/ep['file'],allow_pickle=False) as z:history=z['scene_features'].copy()
  if history.shape!=(ep['observations'],576) or not np.isfinite(history).all():raise ValueError('historical features')
  with np.load(af,allow_pickle=False) as z:W=z['pool_weight'].copy()
  synthetic=[];hashes=[]
  for seed in (879713,879714):
   r=B/f'20261101-flyvis-road-varied-{seed}';rr=read(r/'result.json');file=r/'responses.npz'
   if rr['status']!='completed_causal_four_view_synthetic_road' or sha256(file)!=rr['responses_sha256']:raise ValueError('synthetic source')
   with np.load(file,allow_pickle=False) as z:ret=z['retina'].copy()
   pooled=(np.tanh(ret)@W.T).astype(np.float32);synthetic.append(pooled);hashes.append(sha256(file))
  def summarize(retinal):
   x=torch.tensor(retinal,dtype=torch.float32);centered=x-x.mean(-1,keepdim=True)-boundary.reference;logits=boundary.projection(centered.reshape(-1,64));delta=.5*torch.tanh(logits)
   return {'centered_rms':float(centered.square().mean().sqrt()),'logit_rms':float(logits.square().mean().sqrt()),'logit_abs_p95':float(torch.quantile(logits.abs().flatten(),.95)),'logit_abs_gt2_fraction':float((logits.abs()>2).float().mean()),'delta_rms':float(delta.square().mean().sqrt()),'delta_temporal_std_mean':float(delta.std(0,unbiased=False).mean())},logits.detach().numpy(),delta.detach().numpy()
  rows=[];arrays={}
  histret=history.reshape(-1,4,9,16)[:,:,0,:]
  with torch.no_grad():
   for name,retinal in [('historical_episode06',histret)]+[(f'synthetic_{seed}',arr.reshape(-1,4,16)) for seed,arr in zip((879713,879714),synthetic,strict=True)]:
    metrics,logits,delta=summarize(retinal);rows.append(dict(source=name,count=len(retinal),**metrics));arrays[name+'_logits']=logits;arrays[name+'_delta']=delta
  output=run/'boundary-distributions.npz';np.savez_compressed(output,**arrays)
  result.update(status='completed_pattern_domain_boundary_audit',rows=rows,source_training_admissible=False,historical_episode_sha256=ep['sha256'],synthetic_hashes=hashes,output_sha256=sha256(output),physical_exposure_sync_verified=False,interpretation='Descriptive feature-distribution comparison, pooled synthetic cases include three correlated bend conditions; no ground truth or performance claim')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
