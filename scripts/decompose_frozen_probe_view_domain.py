"""Attribute frozen synthetic probe domain offset to original per-view pooled retina."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();probe=B/'20261101-road-signal-decodability';pr=read(probe/'result.json');wf=probe/'diagnostic.npz';poolfile=B/'20261101-road-scene-pool-audit/pool-audit.npz';src=B/'20261101-joint-vision-yaw-new-sources';sr=read(src/'result.json');prior=B/'20261101-frozen-synthetic-probe-real-history';rr=read(prior/'result.json');dataset=B/'20261029-t1434-body21-flight-cache';dm=read(dataset/'dataset.json')
 if pr['status']!='completed_synthetic_signal_decodability_diagnostic' or sha256(wf)!=pr['outputs_sha256'] or sr['status']!='completed_disjoint_synthetic_sources' or rr['status']!='completed_frozen_synthetic_probe_real_history_diagnostic' or sha256(prior/'scores.npz')!=rr['output_sha256']:raise ValueError('Original diagnostic evidence required')
 with np.load(wf,allow_pickle=False) as z:w=z['pooled_weights'].copy().reshape(4,16);center=z['pooled_center'].copy().reshape(4,16)
 with np.load(poolfile,allow_pickle=False) as z:pool=z['pool_weight'].copy()
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),probe_sha256=sha256(wf),previous_scores_sha256=rr['output_sha256'],actor=False,optimizer_updates=0,scope='fixed probe per-view additive decomposition; no data fit'))
 result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False}
 try:
  synthetic=[];source_hashes=[]
  for seed in (881211,881212):
   root=src/f'validation-{seed}';r=read(root/'result.json');f=root/'responses.npz'
   if sha256(f)!=r['responses_sha256']:raise ValueError('Synthetic source mismatch')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();bends=z['bend'].copy()
   if ret.shape!=(3,41,4,721) or not np.array_equal(bends,[0,-1,1]):raise ValueError('Synthetic retina shape')
   pooled=np.tanh(ret.astype(np.float64))@pool.T;contrib=((pooled-center)*w).sum(axis=-1);synthetic.append(contrib);source_hashes.append(sha256(f))
  syn=np.stack(synthetic);reports=[];arrays={'synthetic_contributions':syn}
  with np.load(prior/'scores.npz',allow_pickle=False) as saved:
   for item in rr['episodes']:
    row=next(x for x in dm['episodes'] if x['file']==item['episode']);f=resolve_payload_path(dataset/row['file']);receipt=read(Path(row['original']['recording'])/'result.json')
    if sha256(f)!=row['sha256'] or receipt.get('training_admissible') is not False:raise ValueError('Original fit-forbidden payload required')
    with load_array(dataset/row['file'],allow_pickle=False) as z:scene=z['scene_features'].copy()
    retina=scene.reshape(-1,4,9,16)[:,:,0,:].astype(np.float64);contrib=((retina-center)*w).sum(axis=-1);score=contrib.sum(axis=1);savedscore=saved[row['file'].replace('.npz','_score')]
    error=float(np.max(np.abs(score-savedscore)))
    if error>1e-10:raise ValueError('Previous probe scores not reproduced')
    means=contrib.mean(axis=0);syn_straight=syn[:,0,-10:,:].mean(axis=(0,1));difference=means-syn_straight
    reports.append({'episode':row['file'],'decisions':len(scene),'score_reproduction_max_abs':error,'view_mean':means.tolist(),'view_std':contrib.std(axis=0).tolist(),'synthetic_straight_reference_mean':syn_straight.tolist(),'difference_from_synthetic_straight':difference.tolist(),'total_mean':float(score.mean()),'absolute_difference_fraction_by_view':(np.abs(difference)/max(1e-12,float(np.abs(difference).sum()))).tolist()});arrays[row['file'].replace('.npz','_contributions')]=contrib
  f=run/'view-contributions.npz';np.savez_compressed(f,**arrays);result.update(status='completed_fixed_probe_view_domain_decomposition',view_order=['front','left','right','rear'],synthetic_sources=source_hashes,reports=reports,output_sha256=sha256(f),fit_on_real=False,physical_or_physiology_verified=False,interpretation='Additive contributions to one frozen synthetic diagnostic classifier, not a physical causal intervention, neural physiology, source attribution of rendering differences, or actor validation.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_fixed_probe_view_domain_decomposition':raise SystemExit(1)
if __name__=='__main__':main()
