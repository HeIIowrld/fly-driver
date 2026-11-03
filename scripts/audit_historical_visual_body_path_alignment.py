"""Audit recorded path-lesion outputs against original diagnostic body histories."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.episode_storage import load_array,resolve_payload_path
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args(); root=B/'20261101-historical-four-episode-path-ablations';data=B/'20261029-t1434-body21-flight-cache';r=read(root/'result.json');manifest=read(data/'dataset.json');dr=read(data/'result.json')
 if r['status']!='completed_future_path_ablation_diagnostic' or r['fit_allowed'] is not False or r['source_training_admissible'] is not False or r['historical_context_missing_all_views'] is not True or sha256(data/'dataset.json')!=dr['dataset_manifest_sha256']:raise ValueError('diagnostic lineage required')
 items={x['file']:x for x in manifest['episodes']};reports=[];arrays={}
 for ep in r['episodes']:
  name=ep['episode'];entry=items[name];cached=resolve_payload_path(data/name);out=root/name.replace('.npz','-ablation-intents.npz');recording=Path(entry['original']['recording']);original=read(recording/'result.json')
  if original.get('training_admissible') is not False or sha256(cached)!=entry['sha256'] or sha256(out)!=ep['output_sha256'] or sha256(recording/'events.jsonl')!=ep['events_sha256']:raise ValueError('fit forbidden and original hashes required')
  with load_array(cached,allow_pickle=False) as z:body=z['body'].copy();valid=z['body_valid'].copy();times=z['observation_time'].copy()
  with np.load(out,allow_pickle=False) as z:
   np.testing.assert_array_equal(times,z['clocks']);base=z['observed'].copy(); spatial=base[:,0]-z['visual_spatial'][:,0];opponent=base[:,0]-z['visual_opponent'][:,0]
  if body.shape!=(len(times),21) or valid.shape!=body.shape or base.shape!=(len(times),2) or not np.isfinite(body).all() or not np.isfinite(base).all() or not np.isfinite(spatial).all() or not np.isfinite(opponent).all():raise ValueError('complete finite aligned history required')
  yaw=body[:,4];speed=body[:,16];usable=valid[:,4]&valid[:,16]&(speed>1);groups={}
  for label,mask in [('positive',usable&(yaw>.02)),('negative',usable&(yaw<-.02))]:
   indices=np.flatnonzero(mask);blocks=int(np.count_nonzero(np.r_[True,np.diff(indices)>1])) if len(indices) else 0
   def summary(x):return {'mean':float(np.mean(x[mask])) if mask.any() else None,'fraction_opposing_yaw':float(np.mean((x[mask]*yaw[mask])<0)) if mask.any() else None}
   groups[label]={'decisions':int(mask.sum()),'contiguous_blocks':blocks,'mean_yaw':float(yaw[mask].mean()) if mask.any() else None,'spatial':summary(spatial),'opponent':summary(opponent)}
  reports.append({'episode':name,'role':entry['role'],'original_fit_allowed':False,'groups':groups,'spatial_full_mean':float(spatial.mean()),'observed_turn_mean':float(base[:,0].mean()),'positive_turn_count':int(np.sum(base[:,0]>0)),'decisions':len(times),'response_sha256':ep['output_sha256'],'source_sha256':entry['sha256']});arrays[name.replace('.npz','')+'_spatial_effect']=spatial;arrays[name.replace('.npz','')+'_opponent_effect']=opponent;arrays[name.replace('.npz','')+'_yaw']=yaw;arrays[name.replace('.npz','')+'_eligible']=usable
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),prior_result_sha256=sha256(root/'result.json'),dataset_manifest_sha256=sha256(data/'dataset.json'),optimizer_updates=0,training_admissible=False));np.savez_compressed(run/'aligned.npz',**arrays)
 result={'status':'completed_historical_visual_path_body_audit','reports':reports,'arrays_sha256':sha256(run/'aligned.npz'),'training_admissible':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'previously examined correlated histories; absent prior-view body context, non-independent blocks; host timestamps are not physical exposure or steering correctness'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
