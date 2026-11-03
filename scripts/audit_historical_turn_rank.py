"""Post-hoc rank diagnostic of frozen outputs against held-out-of-actor future chord."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();src=B/'20261101-historical-four-episode-path-ablations';original=read(src/'result.json');coverage=B/'20261101-historical-future-path-coverage';cr=read(coverage/'result.json');anchorroot=B/'20261101-adaptive-synthetic-neutral-counterfactual-v2';anchor=read(anchorroot/'synthetic-anchor.json');ar=read(anchorroot/'result.json')
 if original['status']!='completed_future_path_ablation_diagnostic' or ar['status']!='completed_adaptive_synthetic_neutral_counterfactual' or anchor['anchor_preactivation']!=ar['synthetic_anchor_preactivation']:raise ValueError('Exact prior results required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_sha256=sha256(src/'result.json'),coverage_sha256=sha256(coverage/'result.json'),synthetic_anchor_sha256=sha256(anchorroot/'synthetic-anchor.json'),scope='posthoc ranking only; fixed synthetic anchor; no threshold fitted on real history',optimizer_updates=0))
 report=[]
 for e in original['episodes']:
  name=e['episode'];f=src/name.replace('.npz','-ablation-intents.npz');path=coverage/name.replace('.npz','-path.npz');match=next(x for x in cr['episodes'] if x['episode']==name)
  if sha256(f)!=e['output_sha256'] or sha256(path)!=match['path_sha256'] or not match['original_fit_forbidden']:raise ValueError('Original source changed')
  with np.load(path,allow_pickle=False) as z:c=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();clocks=z['clock'].copy()
  with np.load(f,allow_pickle=False) as z:
   np.testing.assert_array_equal(clocks,z['clocks']);models={k:z[k][:,0].copy() for k in ('observed','visual_opponent','visual_spatial','added_body_feedback')}
  metrics={}
  for j,h in enumerate((5,10,20)):
   keep=valid[:,j]&(speed>1)&(np.abs(c[:,j])>=.001);left=keep&(c[:,j]<0);right=keep&(c[:,j]>0);metrics[str(h)]={}
   for label,turn in models.items():
    if not np.isfinite(turn).all() or np.any(np.abs(turn)>=1):raise ValueError('Bounded original output required')
    positive=turn[left];negative=turn[right]
    if len(positive) and len(negative):
     pair=positive[:,None]-negative[None,:];auc=float(((pair>0).sum()+.5*(pair==0).sum())/pair.size)
    else:auc=None
    shifted=np.tanh(np.arctanh(turn)-anchor['anchor_preactivation']);ls=float((shifted[left]>0).mean()) if left.any() else None;rs=float((shifted[right]<0).mean()) if right.any() else None
    metrics[str(h)][label]={'left_count':int(left.sum()),'right_count':int(right.sum()),'rank_auc_left_positive':auc,'left_turn_mean':float(positive.mean()) if len(positive) else None,'right_turn_mean':float(negative.mean()) if len(negative) else None,'fixed_synthetic_anchor_left_recall':ls,'fixed_synthetic_anchor_right_recall':rs,'fixed_synthetic_anchor_balanced_accuracy':(ls+rs)/2 if ls is not None and rs is not None else None}
  report.append({'episode':name,'role':e['role'],'horizons':metrics})
 result={'status':'completed_posthoc_rank_diagnostic','episodes':report,'synthetic_anchor_preactivation':anchor['anchor_preactivation'],'fit_allowed':False,'optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False,'interpretation':'Rank AUC is descriptive within each previously examined correlated trajectory and assumes output sign convention. Future driven chord is not steering truth or optimal line. Synthetic anchor is a fixed output-only counterfactual and not deployed.'}
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
