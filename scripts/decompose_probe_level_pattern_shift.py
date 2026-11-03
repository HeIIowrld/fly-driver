"""Separate pooled retina level vs spatial pattern in fixed-probe domain shift."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();prior=B/'20261101-frozen-probe-view-domain-decomposition-v2';r=read(prior/'result.json');probe=B/'20261101-road-signal-decodability';pr=read(probe/'result.json');wf=probe/'diagnostic.npz';poolfile=B/'20261101-road-scene-pool-audit/pool-audit.npz';src=B/'20261101-joint-vision-yaw-new-sources';dataset=B/'20261029-t1434-body21-flight-cache';dm=read(dataset/'dataset.json')
 if r['status']!='completed_fixed_probe_view_domain_decomposition' or sha256(prior/'view-contributions.npz')!=r['output_sha256'] or sha256(wf)!=pr['outputs_sha256']:raise ValueError('Exact previous source required')
 with np.load(wf,allow_pickle=False) as z:w=z['pooled_weights'].copy().reshape(4,16);center=z['pooled_center'].copy().reshape(4,16)
 with np.load(poolfile,allow_pickle=False) as z:pool=z['pool_weight'].copy()
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(prior/'result.json'),probe_sha256=sha256(wf),actor=False,optimizer_updates=0));result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False}
 try:
  synth=[]
  for seed in (881211,881212):
   root=src/f'validation-{seed}';rr=read(root/'result.json');f=root/'responses.npz'
   if sha256(f)!=rr['responses_sha256']:raise ValueError('Synthetic source mismatch')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'][0,-10:].copy()
   synth.append(np.tanh(ret.astype(np.float64))@pool.T)
  synthetic_mean=np.concatenate(synth,axis=0).mean(axis=0);reports=[];arrays={'synthetic_view_mean':synthetic_mean,'weights':w}
  for entry in r['reports']:
   row=next(x for x in dm['episodes'] if x['file']==entry['episode']);f=resolve_payload_path(dataset/row['file']);receipt=read(Path(row['original']['recording'])/'result.json')
   if sha256(f)!=row['sha256'] or receipt.get('training_admissible') is not False:raise ValueError('Exact fit-forbidden history required')
   with load_array(dataset/row['file'],allow_pickle=False) as z:scene=z['scene_features'].copy()
   real=scene.reshape(-1,4,9,16)[:,:,0,:].astype(np.float64);real_mean=real.mean(axis=0);delta=real_mean-synthetic_mean;level=delta.mean(axis=1);uniform=level*w.sum(axis=1);pattern=((delta-level[:,None])*w).sum(axis=1);total=uniform+pattern
   np.testing.assert_allclose(total,entry['difference_from_synthetic_straight'],atol=1e-9,rtol=0)
   reports.append({'episode':row['file'],'real_pooled_level_by_view':real_mean.mean(axis=1).tolist(),'synthetic_pooled_level_by_view':synthetic_mean.mean(axis=1).tolist(),'uniform_level_contribution_delta':uniform.tolist(),'spatial_pattern_contribution_delta':pattern.tolist(),'total_probe_contribution_delta':total.tolist()});arrays[row['file'].replace('.npz','_delta')]=delta
  out=run/'level-pattern.npz';np.savez_compressed(out,**arrays);result.update(status='completed_probe_level_pattern_decomposition',view_order=['front','left','right','rear'],reports=reports,output_sha256=sha256(out),fit_on_real=False,interpretation='Exact algebra in pooled, tanh-transformed retina space of one fixed synthetic probe; uniform level is not proof of illumination and spatial pattern is not a causal rendering diagnosis. Previously examined data, no labels used.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_probe_level_pattern_decomposition':raise SystemExit(1)
if __name__=='__main__':main()
