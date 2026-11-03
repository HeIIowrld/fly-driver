"""Describe historical forecast error by recorded capture repetition, never fit."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 root=B/'20261029-t1434-body21-flight-cache';manifest=read(root/'dataset.json');sr=read(root/'result.json')
 if sha256(root/'dataset.json')!=sr['dataset_manifest_sha256']:raise ValueError('manifest hash')
 gains={}; hashes={}
 for name,folder in [('original','20261101-causal-motion-forecast-synthetic-fit'),('reversal','20261101-reversing-yaw-forecast-synthetic-fit')]:
  fit=B/folder; result=read(fit/'result.json'); ck=fit/'motion-forecast.pt'
  if sha256(ck)!=result['checkpoint_sha256'] or result['actor_qualified'] is not False:raise ValueError('checkpoint')
  state=torch.load(ck,map_location='cpu',weights_only=True)
  if state['actor_qualified'] is not False:raise ValueError('checkpoint qualification')
  gains[name]=state['model']['raw_gain'].tanh().numpy().reshape(1,1,8,1);hashes[name]=sha256(ck)
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),dataset_sha256=sha256(root/'dataset.json'),checkpoint_sha256=hashes,scope='posthoc capture identity stratification, no fitting or actor'))
 groups={k:[] for k in ['all','both_new','previous_repeat','next_repeat','both_repeat']};episodes=[]
 for e in manifest['episodes']:
  f=resolve_payload_path(root/e['file'])
  if sha256(f)!=e['sha256']:raise ValueError('payload hash')
  rec=Path(e['original']['recording']);result=read(rec/'result.json')
  if result.get('training_admissible') is not False or sha256(rec/'events.jsonl')!=e['source_events_sha256']:raise ValueError('source receipt')
  with load_array(root/e['file'],allow_pickle=False) as z:scene=z['scene_features'].copy();clock=z['observation_time'].copy()
  with (rec/'events.jsonl').open(encoding='utf8') as stream:events=[json.loads(line) for line in stream]
  if scene.shape!=(len(clock),576) or len(events)!=len(clock) or not np.isfinite(scene).all() or not np.isfinite(clock).all():raise ValueError('history shape')
  if any(ev['index']!=i or abs(ev['decision_monotonic_seconds']-clock[i])>1e-6 for i,ev in enumerate(events)):raise ValueError('unmatched event order or clock')
  ids=[(x['camera']['generation'],x['camera']['capture_id']) for x in events];times=np.array([x['camera']['received'] for x in events]);rgb=[[v['rgb_sha256'] for v in x['camera']['views']] for x in events]
  if any([v['view_id'] for v in x['camera']['views']]!=['front','left','right','rear'] for x in events):raise ValueError('view order')
  if not np.isfinite(times).all() or np.any(np.diff(times)<0):raise ValueError('camera times')
  old=np.array([ids[i]==ids[i-1] for i in range(1,len(ids))]);delta=np.diff(clock); eligible=(delta[:-1]>=.036)&(delta[:-1]<=.044)&(delta[1:]>=.036)&(delta[1:]<=.044)
  x=scene.reshape(-1,4,9,16)[:,:,1:];prev=(x[1:-1]-x[:-2]).astype('float64');target=(x[2:]-x[1:-1]).astype('float64')
  hold=np.square(target).mean(axis=(1,2,3));errors={n:np.square(prev*g-target).mean(axis=(1,2,3)) for n,g in gains.items()}
  masks={'all':eligible,'both_new':eligible & ~old[:-1] & ~old[1:],'previous_repeat':eligible & old[:-1] & ~old[1:],'next_repeat':eligible & ~old[:-1] & old[1:],'both_repeat':eligible & old[:-1] & old[1:]}
  for name,mask in masks.items():groups[name].append(dict(count=int(mask.sum()),hold_sum=float(hold[mask].sum()),**{n+'_sum':float(err[mask].sum()) for n,err in errors.items()}))
  repeated_rgb=[sum(rgb[i][j]==rgb[i-1][j] for i in range(1,len(rgb))) for j in range(4)]
  unique_gap=np.diff(times)[~old]; episodes.append(dict(file=e['file'],capture_repeat_decisions=int(old.sum()),rgb_equal_adjacent_counts=repeated_rgb,unique_capture_host_gap_p50_ms=float(np.median(unique_gap)*1000),unique_capture_host_gap_p95_ms=float(np.percentile(unique_gap,95)*1000),unique_capture_host_gap_max_ms=float(unique_gap.max()*1000)))
 def summarize(rows):
  n=sum(r['count'] for r in rows);h=sum(r['hold_sum'] for r in rows);o={'windows':n,'hold_mse':h/n if n else None}
  for name in gains:
   s=sum(r[name+'_sum'] for r in rows);o[name+'_mse']=s/n if n else None;o[name+'_relative_change']=s/h-1 if h else None
  return o
 record=dict(status='completed_capture_repeat_forecast_audit',groups={k:summarize(v) for k,v in groups.items()},episodes=episodes,interpretation='Recorded capture IDs and host receipt time, not physical exposure timing; future capture identity defines evaluation strata only. Windows correlated; no independent trials.',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 write_json(run/'result.json',record); print(json.dumps(record,allow_nan=False))
if __name__=='__main__':main()
