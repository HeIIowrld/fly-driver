"""Compare fixed global/per-view contrast gain on fit-forbidden real scenes."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();dataset=B/'20261029-t1434-body21-flight-cache';dm=read(dataset/'dataset.json');original=B/'20261101-historical-four-episode-path-ablations';r=read(original/'result.json');ck=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';cr=read(ck.parent/'result.json')
 if r['status']!='completed_future_path_ablation_diagnostic' or sha256(ck)!=cr['checkpoint_sha256']:raise ValueError('Original diagnostic sources required')
 c=torch.load(ck,map_location='cpu',weights_only=True);reference=c['reference'].numpy();weight=c['weight'].numpy();limit=float(c['max_contrast_rms']);run=begin_run(a.output,dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(ck),per_view_source_sha256=sha256('src/flybrain_drive/per_view_retinal_contrast_boundary.py'),actor=False,optimizer_updates=0));result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False}
 try:
  reports=[];arrays={}
  for ep in r['episodes']:
   row=next(x for x in dm['episodes'] if x['file']==ep['episode']);f=resolve_payload_path(dataset/row['file']);receipt=read(Path(row['original']['recording'])/'result.json')
   if sha256(f)!=row['sha256'] or receipt['training_admissible'] is not False:raise ValueError('Original fit-forbidden source required')
   with load_array(dataset/row['file'],allow_pickle=False) as z:scene=z['scene_features'].copy()
   retina=scene.reshape(-1,4,9,16)[:,:,0,:];center=retina-retina.mean(axis=-1,keepdims=True)-reference;global_rms=np.sqrt(np.mean(center**2,axis=(1,2),keepdims=True)+1e-24);view_rms=np.sqrt(np.mean(center**2,axis=2,keepdims=True)+1e-24);gg=np.minimum(1,limit/global_rms);vg=np.minimum(1,limit/view_rms);global_logits=(center*gg).reshape(-1,64)@weight.T;per_logits=(center*vg).reshape(-1,64)@weight.T
   reports.append({'episode':row['file'],'decisions':len(scene),'global_gain_median':float(np.median(gg)),'per_view_gain_median':np.median(vg,axis=0).ravel().tolist(),'global_abs_logit_gt2_fraction':float(np.mean(np.abs(global_logits)>2)),'per_view_abs_logit_gt2_fraction':float(np.mean(np.abs(per_logits)>2)),'global_delta_rms':float(np.sqrt(np.mean((.5*np.tanh(global_logits))**2))),'per_view_delta_rms':float(np.sqrt(np.mean((.5*np.tanh(per_logits))**2)))})
   arrays[row['file'].replace('.npz','_global_logits')]=global_logits;arrays[row['file'].replace('.npz','_per_view_logits')]=per_logits
  out=run/'logits.npz';np.savez_compressed(out,**arrays);result.update(status='completed_real_scene_per_view_limit_diagnostic',reports=reports,output_sha256=sha256(out),fit_on_real=False,training_admissible=False,interpretation='Fixed original synthetic threshold and weights on fit-forbidden real pooled scene; no optimization, physical sensor or actor claim. Per-view scaling can change effective contribution and does not itself prove correct direction.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_real_scene_per_view_limit_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
