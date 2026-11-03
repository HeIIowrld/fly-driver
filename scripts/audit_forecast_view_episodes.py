"""Offline stratified diagnostics for frozen synthetic forecasts; no fitting or vehicle control."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.episode_storage import load_array, resolve_payload_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

BASE = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a=p.parse_args()
 root=BASE/'20261029-t1434-body21-flight-cache'; manifest=read(root/'dataset.json'); source=read(root/'result.json')
 if sha256(root/'dataset.json') != source['dataset_manifest_sha256']: raise ValueError('manifest hash mismatch')
 configs=[('original','20261101-causal-motion-forecast-synthetic-fit'),('reversal','20261101-reversing-yaw-forecast-synthetic-fit')]
 gains={}; hashes={}
 for name,folder in configs:
  d=BASE/folder; result=read(d/'result.json'); f=d/'motion-forecast.pt'
  if sha256(f)!=result['checkpoint_sha256'] or result['actor_qualified'] is not False: raise ValueError('checkpoint mismatch or qualification')
  data=torch.load(f,map_location='cpu',weights_only=True)
  if data['actor_qualified'] is not False: raise ValueError('unexpected qualification')
  gains[name]=data['model']['raw_gain'].tanh().numpy().reshape(1,1,8,1); hashes[name]=sha256(f)
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),dataset_sha256=sha256(root/'dataset.json'),checkpoint_sha256=hashes,scope='posthoc evaluation only; no fitting',vehicle_control=False))
 rows=[]; totals={v:{m:[0,0.,0.] for m in gains} for v in ('front','left','right','rear')}
 for e in manifest['episodes']:
  if sha256(resolve_payload_path(root/e['file'])) != e['sha256']: raise ValueError('episode hash mismatch')
  rec=read(Path(e['original']['recording'])/'result.json')
  if rec.get('training_admissible') is not False: raise ValueError('unexpected source permission')
  with load_array(root/e['file'],allow_pickle=False) as z: scene=z['scene_features'].copy(); clock=z['observation_time'].copy()
  if scene.shape!=(len(clock),576) or not np.isfinite(scene).all() or not np.isfinite(clock).all() or np.any(np.diff(clock)<=0): raise ValueError('invalid observations')
  dt=np.diff(clock); good=(dt[:-1]>=.036)&(dt[:-1]<=.044)&(dt[1:]>=.036)&(dt[1:]<=.044)
  x=scene.reshape(-1,4,9,16)[:,:,1:]; previous=(x[1:-1]-x[:-2])[good].astype(np.float64); future=(x[2:]-x[1:-1])[good].astype(np.float64)
  hold=np.square(future).mean(axis=(2,3)); row={'file':e['file'],'windows':int(good.sum()),'views':{}}
  for j,v in enumerate(totals):
   h=hold[:,j]; prev_energy=np.square(previous[:,j]).mean(axis=(1,2)); nxt_energy=h
   item={'hold_mse':float(h.mean()),'previous_exact_zero_windows':int((prev_energy==0).sum()),'next_exact_zero_windows':int((nxt_energy==0).sum())}
   for name,gain in gains.items():
    err=np.square(previous[:,j]*gain[:,0]-future[:,j]).mean(axis=(1,2)); item[name+'_mse']=float(err.mean()); item[name+'_relative_change']=float(err.sum()/h.sum()-1) if h.sum() else None
    t=totals[v][name]; t[0]+=len(h);t[1]+=float(err.sum());t[2]+=float(h.sum())
   row['views'][v]=item
  rows.append(row)
 aggregate={v:{m:dict(windows=t[0],predictor_mse=t[1]/t[0],hold_mse=t[2]/t[0],relative_change=t[1]/t[2]-1 if t[2] else None) for m,t in models.items()} for v,models in totals.items()}
 result=dict(status='completed_view_episode_stratified_forecast_audit',episodes=rows,views=aggregate,interpretation='Posthoc per-view errors on previously examined, temporally correlated histories; fit forbidden, no independent trials. View names follow the dataset order and do not establish physical camera calibration.',optimizer_updates=0,actor_deployed=False,physical_timing_verified=False,vehicle_control_sent=False)
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
