"""Offline synthetic-neutral output counterfactual on a fixed historical rollout."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 n=B/'20261101-unified-synthetic-neutral-stability';nr=read(n/'result.json');nf=n/'outputs.npz';coverage=B/'20261101-historical-future-path-coverage';cr=read(coverage/'result.json');ep=next(x for x in cr['episodes'] if x['episode']=='episode-06.npz');path=coverage/'episode-06-path.npz';rrdir=B/'20261101-multispeed-yaw-coverage-selected';rr=read(rrdir/'result.json');rf=rrdir/'episode-06-body-ablation-intents.npz'
 if nr['status']!='completed_unified_synthetic_neutral_stability' or sha256(nf)!=nr['outputs_sha256'] or ep['original_fit_forbidden'] is not True or sha256(path)!=ep['path_sha256'] or rr['status']!='completed_single_graph_multispeed_yaw_coverage_selected_replay' or sha256(rf)!=rr['episodes'][0]['output_sha256'] or ep['events_sha256']!=rr['episodes'][0]['source_events_sha256']:raise ValueError('exact sources and role required')
 with np.load(nf,allow_pickle=False) as z:anchor=z['intents'][0,100].copy();counter=z['neutral_counterfactual'][0,100].copy()
 if not np.allclose(anchor,nr['conditions']['anchor']['original_at_101'],rtol=0,atol=1e-7) or not np.array_equal(counter,np.zeros(2)):raise ValueError('synthetic anchor mismatch')
 with np.load(path,allow_pickle=False) as z:curve=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();clock=z['clock'].copy()
 with np.load(rf,allow_pickle=False) as z:values=z['intents'].copy();np.testing.assert_array_equal(clock,z['clocks'])
 if values.shape!=(2,len(clock),2) or np.any(abs(values)>=1) or np.any(abs(anchor)>=1):raise ValueError('finite inverse tanh required')
 adjusted=np.tanh(np.arctanh(values)-np.arctanh(anchor))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),neutral_sha256=sha256(nf),path_sha256=sha256(path),intent_sha256=sha256(rf),reference='existing synthetic zero body 101st decision, no historical target tuning',scope='posthoc output-only counterfactual; not calibration or control'))
 reports={}
 for j,h in enumerate((5,10,20)):
  keep=valid[:,j]&(speed>1)&(abs(curve[:,j])>=.001);truth=curve[keep,j];left=truth<0;right=truth>0
  if not left.any() or not right.any():raise ValueError('both direction samples needed')
  def score(v):
   x=v[keep,0];pred=np.sign(x);ok=(pred==-np.sign(truth));rank=rankdata(x);auc=(rank[left].sum()-left.sum()*(left.sum()+1)/2)/(left.sum()*right.sum());l=float(ok[left].mean());r=float(ok[right].mean())
   return dict(accuracy=float(ok.mean()),balanced_accuracy=(l+r)/2,left_recall=l,right_recall=r,positive=int((x>0).sum()),negative=int((x<0).sum()),left_vs_right_auc=float(auc))
  reports[str(h)]=dict(n=int(keep.sum()),left=int(left.sum()),right=int(right.sum()),constant_majority_accuracy=float(max(left.sum(),right.sum())/len(truth)),original=score(values[1]),synthetic_neutral_counterfactual=score(adjusted[1]))
 output=run/'outputs.npz';np.savez_compressed(output,original=values,synthetic_neutral=adjusted,clock=clock)
 result=dict(status='completed_synthetic_neutral_historical_counterfactual',synthetic_anchor=anchor.tolist(),horizons=reports,outputs_sha256=sha256(output),interpretation='Synthetic-only anchor is not qualified for real vehicle; atanh shift assumes same readout gain/mix and affects output only, graph is not rerun. Previously examined, coverage-selected correlated trajectory; future chord is not direct steering target.',fit_allowed=False,actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0,physical_timing_verified=False)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
