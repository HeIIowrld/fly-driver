"""Frozen synthetic retina probe on fit-forbidden recordings; never actor input."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();probe=B/'20261101-road-signal-decodability';pr=read(probe/'result.json');weights=probe/'diagnostic.npz';poolfile=B/'20261101-road-scene-pool-audit/pool-audit.npz';src=B/'20261101-joint-vision-yaw-new-sources';sr=read(src/'result.json');fit=B/'20261101-historical-four-episode-path-ablations';fr=read(fit/'result.json');data=B/'20261029-t1434-body21-flight-cache';dm=read(data/'dataset.json')
 if pr['status']!='completed_synthetic_signal_decodability_diagnostic' or sha256(weights)!=pr['outputs_sha256'] or sr['status']!='completed_disjoint_synthetic_sources' or fr['status']!='completed_future_path_ablation_diagnostic':raise ValueError('Exact original frozen sources required')
 with np.load(weights,allow_pickle=False) as z:w=z['pooled_weights'].copy();center=z['pooled_center'].copy()
 with np.load(poolfile,allow_pickle=False) as z:pool=z['pool_weight'].copy()
 if w.shape!=(64,) or center.shape!=(64,) or pool.shape!=(16,721):raise ValueError('Original four-view feature contract required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),probe_sha256=sha256(weights),prior_run_sha256=sha256(fit/'result.json'),synthetic_sources_sha256=sha256(src/'result.json'),actor=False,optimizer_updates=0,scope='fixed synthetic diagnostic probe only, no real-data fit or actor input'))
 result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False}
 try:
  synthetic=[]
  for seed in (881211,881212):
   root=src/f'validation-{seed}';rr=read(root/'result.json');f=root/'responses.npz'
   if sha256(f)!=rr['responses_sha256']:raise ValueError('Synthetic response changed')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();bends=z['bend'].copy()
   if ret.shape!=(3,41,4,721) or not np.array_equal(bends,[0,-1,1]):raise ValueError('Original synthetic cases required')
   pooled=(np.tanh(ret.astype(np.float64))@pool.T).reshape(3,41,64);scores=(pooled-center)@w
   synthetic.append({'seed':seed,'bend_minus_final_score':float(scores[1,-1]),'bend_plus_final_score':float(scores[2,-1]),'bend_plus_minus_final':float(scores[2,-1]-scores[1,-1]),'pair_ranking_over_last10':float(np.mean(scores[2,-10:]>scores[1,-10:]))})
  saved={};recordings=[]
  for ep in fr['episodes']:
   row=next(x for x in dm['episodes'] if x['file']==ep['episode']);payload=resolve_payload_path(data/row['file']);recording=Path(row['original']['recording']);receipt=read(recording/'result.json')
   if receipt['training_admissible'] is not False or sha256(payload)!=row['sha256'] or sha256(recording/'events.jsonl')!=ep['events_sha256']:raise ValueError('Exact original fit-forbidden history required')
   with load_array(data/row['file'],allow_pickle=False) as z:scene=z['scene_features'].copy();clock=z['observation_time'].copy()
   if scene.shape!=(len(clock),576) or not np.isfinite(scene).all():raise ValueError('Original pooled scene required')
   retinal=scene.reshape(-1,4,9,16)[:,:,0,:].reshape(-1,64);score=(retinal.astype(np.float64)-center)@w
   # No privileged labels are read until all frozen probe scores are computed.
   saved[row['file']]=(score,clock);recordings.append(row['file'])
  coverage=B/'20261101-historical-future-path-coverage';coverage_result=read(coverage/'result.json');reports=[];arrays={}
  for ep in fr['episodes']:
   name=ep['episode'];score,clock=saved[name];path=coverage/name.replace('.npz','-path.npz');er=next(x for x in coverage_result['episodes'] if x['episode']==name)
   if sha256(path)!=er['path_sha256'] or er['events_sha256']!=ep['events_sha256'] or er['original_fit_forbidden'] is not True:raise ValueError('Exact posthoc path required')
   with np.load(path,allow_pickle=False) as z:c=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();np.testing.assert_array_equal(clock,z['clock'])
   output=fit/name.replace('.npz','-ablation-intents.npz')
   if sha256(output)!=ep['output_sha256']:raise ValueError('Original graph replay changed')
   with np.load(output,allow_pickle=False) as z:turn=z['observed'][:,0].copy();np.testing.assert_array_equal(clock,z['clocks'])
   horizons={}
   for j,h in enumerate((5,10,20)):
    mask=valid[:,j]&(speed>1)&(np.abs(c[:,j])>=.001);left=mask&(c[:,j]<0);right=mask&(c[:,j]>0)
    def auc(x):
     if not left.any() or not right.any():return None
     pair=x[left,None]-x[None,right];return float(((pair>0).sum()+.5*(pair==0).sum())/pair.size)
    horizons[str(h)]={'left':int(left.sum()),'right':int(right.sum()),'probe_auc_left_positive_assumed':auc(score),'graph_auc_left_positive_assumed':auc(turn),'probe_score_left_mean':float(score[left].mean()) if left.any() else None,'probe_score_right_mean':float(score[right].mean()) if right.any() else None,'probe_positive':int((score[mask]>0).sum()),'probe_negative':int((score[mask]<0).sum())}
   reports.append({'episode':name,'role':ep['role'],'horizons':horizons});arrays[name.replace('.npz','_score')]=score
  out=run/'scores.npz';np.savez_compressed(out,**arrays);result.update(status='completed_frozen_synthetic_probe_real_history_diagnostic',synthetic_validation=synthetic,episodes=reports,output_sha256=sha256(out),fit_on_real=False,actor_input_future_truth=False,training_admissible=False,physical_or_physiology_verified=False,interpretation='Frozen synthetic-only comparator applied posthoc to previously examined correlated real histories. Future chord and sign mapping are diagnostic assumptions; not a driver, independent holdout, biological or performance proof.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_frozen_synthetic_probe_real_history_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
