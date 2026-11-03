"""Check temporal and speed confounding in previously examined path/intent association."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def auc(x,label):
 n=int(label.sum());m=len(label)-n
 if not n or not m:return None
 ranks=rankdata(x);return float((ranks[label].sum()-n*(n+1)/2)/(n*m))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=B/'20261101-historical-future-path-coverage';cr=read(root/'result.json');e=next(x for x in cr['episodes'] if x['episode']=='episode-06.npz');pf=root/'episode-06-path.npz';rep=B/'20261101-multispeed-yaw-coverage-selected';rr=read(rep/'result.json');out=rep/'episode-06-body-ablation-intents.npz'
 if e['original_fit_forbidden'] is not True or sha256(pf)!=e['path_sha256'] or sha256(out)!=rr['episodes'][0]['output_sha256'] or rr['episodes'][0]['source_events_sha256']!=e['events_sha256']:raise ValueError('exact fit-forbidden source')
 with np.load(pf,allow_pickle=False) as z:curve=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();clock=z['clock'].copy()
 with np.load(out,allow_pickle=False) as z:intents=z['intents'].copy();np.testing.assert_array_equal(clock,z['clocks'])
 keep=valid[:,2]&(speed>1)&(abs(curve[:,2])>=.001);indices=np.flatnonzero(keep);y=curve[keep,2]<0;x=intents[1,keep,0];t=clock[keep]-clock[keep][0];v=speed[keep];transitions=np.flatnonzero(y[1:]!=y[:-1])+1
 bins=[]
 for lo,hi in ((1,3),(3,5),(5,7),(7,9),(9,12)):
  m=(v>=lo)&(v<hi);bins.append(dict(speed_m_s=[lo,hi],count=int(m.sum()),left=int(y[m].sum()),right=int((~y[m]).sum()),auc_left_greater=auc(x[m],y[m])))
 left_times=t[y];right_times=t[~y]
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),path_sha256=sha256(pf),intent_sha256=sha256(out),scope='posthoc confounding diagnostic, no model fitting or actor access'))
 result=dict(status='completed_temporal_confound_diagnostic',rows=len(x),left=int(y.sum()),right=int((~y).sum()),direction_transitions=int(len(transitions)),transition_original_indices=indices[transitions].tolist(),left_time_range_seconds=[float(left_times.min()),float(left_times.max())],right_time_range_seconds=[float(right_times.min()),float(right_times.max())],spearman_time_vs_intent=float(spearmanr(t,x).statistic),spearman_speed_vs_intent=float(spearmanr(v,x).statistic),auc_left_greater=auc(x,y),speed_bins=bins,interpretation='Time, speed and path truth are diagnostic only; strong temporal dependence means samples and pairs are not independent. No physiological or driving conclusion.',optimizer_updates=0,fit_allowed=False,actor_deployed=False,vehicle_control_sent=False)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
