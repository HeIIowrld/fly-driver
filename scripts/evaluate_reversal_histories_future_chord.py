"""Compare selected direction-reversal histories; evaluation only, no fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();cov=B/'20261101-historical-future-path-coverage';cm=read(cov/'result.json');rep=B/'20261101-multispeed-yaw-reversal-histories';rr=read(rep/'result.json')
 if rr['status']!='completed_single_graph_multispeed_yaw_reversal_histories_replay':raise ValueError('replay')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),coverage_sha256=sha256(cov/'result.json'),replay_sha256=sha256(rep/'result.json'),scope='posthoc two selected histories; future truth evaluation only, never fit'))
 results=[]
 for name in ('episode-03.npz','episode-04.npz'):
  ep=next(x for x in cm['episodes'] if x['episode']==name);re=next(x for x in rr['episodes'] if x['episode']==name);pf=cov/name.replace('.npz','-path.npz');rf=rep/name.replace('.npz','-body-ablation-intents.npz')
  if ep['original_fit_forbidden'] is not True or re['source_training_admissible'] is not False or sha256(pf)!=ep['path_sha256'] or sha256(rf)!=re['output_sha256'] or ep['events_sha256']!=re['source_events_sha256']:raise ValueError('source provenance')
  with np.load(pf,allow_pickle=False) as z:curve=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();clock=z['clock'].copy()
  with np.load(rf,allow_pickle=False) as z:x=z['intents'].copy();np.testing.assert_array_equal(clock,z['clocks'])
  m=valid[:,2]&(speed>1)&(abs(curve[:,2])>=.001);truth=curve[m,2];left=truth<0;right=truth>0
  if not left.any() or not right.any():raise ValueError('both directions')
  values=x[1,m,0];pred=np.sign(values);ok=pred==-np.sign(truth);rank=rankdata(values);auc=float((rank[left].sum()-left.sum()*(left.sum()+1)/2)/(left.sum()*right.sum()));t=clock[m]-clock[m][0]
  results.append(dict(episode=name,rows=len(truth),left=int(left.sum()),right=int(right.sum()),direction_transitions=int(np.sum(left[1:]!=left[:-1])),positive=int((values>0).sum()),negative=int((values<0).sum()),accuracy=float(ok.mean()),balanced_accuracy=float((ok[left].mean()+ok[right].mean())/2),constant_right_accuracy=float(right.mean()),left_vs_right_auc=auc,spearman_time_vs_intent=float(spearmanr(t,values).statistic),fit_allowed=False))
 result=dict(status='completed_reversal_history_diagnostic',episodes=results,interpretation='Only two selected, previously examined trajectories with temporal dependence; future driven chord is not immediate steering or optimal line. Physical signal timing and sign conventions remain unverified.',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
