"""Read-only evaluation of synthetic contrast limit on one fit-forbidden history."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();fit=B/'20261101-adaptive-pattern-fit';r=read(fit/'result.json');ck=fit/'adaptive-pattern-contrast.pt';root=B/'20261029-t1434-body21-flight-cache';manifest=root/'dataset.json'
 if r['status']!='completed_synthetic_adaptive_pattern_fit' or sha256(ck)!=r['checkpoint_sha256'] or sha256(manifest)!=read(root/'result.json')['dataset_manifest_sha256']:raise ValueError('source mismatch')
 ep=next(e for e in read(manifest)['episodes'] if e['file']=='episode-06.npz');record=Path(ep['original']['recording'])
 if read(record/'result.json').get('training_admissible') is not False or sha256(resolve_payload_path(root/ep['file']))!=ep['sha256']:raise ValueError('fit forbidden history required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(ck),episode_sha256=ep['sha256'],optimizer_updates=0))
 try:
  p=torch.load(ck,map_location='cpu',weights_only=True)
  if p['actor_qualified'] is not False or p['max_contrast_rms']<=0:raise ValueError('research checkpoint required')
  with load_array(root/ep['file'],allow_pickle=False) as z:x=z['scene_features'].copy().reshape(-1,4,9,16)[:,:,0,:]
  c=x-x.mean(-1,keepdims=True)-p['reference'].numpy();rms=np.sqrt(np.mean(c*c,axis=(1,2)));gain=np.minimum(1.,p['max_contrast_rms']/np.maximum(rms,1e-12));logits=c.reshape(-1,64)@p['weight'].numpy().T
  result=dict(status='completed_adaptive_contrast_historical_audit',limit_from_synthetic_only=p['max_contrast_rms'],gain_min=float(gain.min()),gain_median=float(np.median(gain)),gain_max=float(gain.max()),pre_limit_logit_abs_gt2=float(np.mean(np.abs(logits)>2)),post_limit_logit_abs_gt2=float(np.mean(np.abs(logits*gain[:,None])>2)),post_limit_logit_rms=float(np.sqrt(np.mean((logits*gain[:,None])**2))),source_training_admissible=False,fit_or_selection_on_historical=False,optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 except Exception as e:result=dict(status='failed',reason=str(e),error=type(e).__name__)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
