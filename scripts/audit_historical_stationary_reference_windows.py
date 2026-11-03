"""Check whether archived diagnostic episodes contain a causal stationary reference window."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.episode_storage import load_array,resolve_payload_path
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=B/'20261029-t1434-body21-flight-cache';manifest=read(root/'dataset.json');rr=read(root/'result.json')
 if sha256(root/'dataset.json')!=rr['dataset_manifest_sha256']:raise ValueError('original manifest mismatch')
 reports=[]
 for ep in manifest['episodes']:
  if ep['file'] not in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):continue
  f=resolve_payload_path(root/ep['file']);rec=Path(ep['original']['recording']);receipt=read(rec/'result.json')
  if sha256(f)!=ep['sha256'] or receipt.get('training_admissible') is not False:raise ValueError('original fit-forbidden data required')
  with load_array(f,allow_pickle=False) as z:body=z['body'].copy();valid=z['body_valid'].copy();times=z['observation_time'].copy()
  ids=[2,3,4,16,17,18,19,20];available=valid[:,ids].all(axis=1);slow=body[:,16]<=.1/3.6;wheel=(np.abs(body[:,17:21])<=.05).all(axis=1);angular=(np.abs(body[:,2:5])<=.02).all(axis=1);ok=available&slow&wheel&angular;idx=np.flatnonzero(ok);blocks=np.split(idx,np.flatnonzero(np.diff(idx)>1)+1);qual=[x for x in blocks if len(x)>=11 and times[x[-1]]-times[x[0]]>=.4-1e-9];reports.append({'episode':ep['file'],'decisions':len(times),'available':int(available.sum()),'speed_condition':int((available&slow).sum()),'wheel_condition':int((available&wheel).sum()),'angular_condition':int((available&angular).sum()),'all_stationary_conditions':int(ok.sum()),'qualified_windows':len(qual),'source_sha256':ep['sha256']})
 if len(reports)!=4:raise ValueError('four original episodes required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),dataset_sha256=sha256(root/'dataset.json'),max_speed_kmh=.1,max_wheel_rate=.05,max_angular_rate=.02,min_frames=11,min_span_s=.4,optimizer_updates=0));result={'status':'completed_historical_stationary_window_audit','reports':reports,'training_admissible':False,'vehicle_control_sent':False,'caveat':'strict existing stationary reference thresholds; absence does not prove vehicle was never stopped'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
