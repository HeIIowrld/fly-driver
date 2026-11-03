"""Score a posthoc coverage-selected history against offline future chord and constant baselines."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 coverage=B/'20261101-historical-future-path-coverage';cr=read(coverage/'result.json');row=next(e for e in cr['episodes'] if e['episode']=='episode-06.npz');path=coverage/'episode-06-path.npz'
 replay=B/'20261101-adaptive-selected-history-replay';rr=read(replay/'result.json');out=replay/'episode-06-body-ablation-intents.npz'
 if cr['status']!='completed_historical_future_path_coverage' or row['original_fit_forbidden'] is not True or sha256(path)!=row['path_sha256'] or rr['status']!='completed_synthetic_adaptive_selected_history_replay' or sha256(out)!=rr['episodes'][0]['output_sha256'] or rr['episodes'][0]['source_events_sha256']!=row['events_sha256'] or rr['episodes'][0]['source_training_admissible'] is not False:raise ValueError('source and role')
 with np.load(path,allow_pickle=False) as z:c=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();clock=z['clock'].copy()
 with np.load(out,allow_pickle=False) as z:intent=z['intents'].copy();np.testing.assert_array_equal(clock,z['clocks'])
 if intent.shape!=(2,len(clock),2):raise ValueError('original paired output')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),coverage_sha256=sha256(coverage/'result.json'),path_sha256=sha256(path),replay_sha256=sha256(out),scope='posthoc selection, privileged truth for evaluation only, not independent or training'))
 reports={}
 for j,h in enumerate((5,10,20)):
  m=valid[:,j]&(speed>1)&(np.abs(c[:,j])>=.001);truth=c[m,j];target=-np.sign(truth);left=truth<0;right=truth>0
  if not left.any() or not right.any():raise ValueError('two direction check')
  models=[]
  for v in intent[:,m,0]:
   ok=np.sign(v)==target;l=float(ok[left].mean());r=float(ok[right].mean());models.append(dict(match=int(ok.sum()),total=len(ok),accuracy=float(ok.mean()),left_recall=l,right_recall=r,balanced_accuracy=(l+r)/2,positive_intents=int((v>0).sum()),negative_intents=int((v<0).sum())))
  reports[str(h)]=dict(left=int(left.sum()),right=int(right.sum()),constant_right_accuracy=float(right.mean()),constant_right_balanced_accuracy=.5,original=models[0],adaptive_candidate=models[1])
 result=dict(status='completed_adaptive_selected_history_future_chord_score',horizons=reports,interpretation='Future driven chord is not immediate steering or racing line; model sign convention is assumed, physical sensor/actuator timing not established. One coverage-selected, previously examined, temporally correlated trajectory.',fit_allowed=False,actor_input_future_truth=False,optimizer_updates=0,vehicle_control_sent=False,actor_deployed=False)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
