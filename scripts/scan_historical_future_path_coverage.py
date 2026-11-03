"""Offline coverage inventory of historical future path directions; never fit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_geometry import camera_geometry,matrix
from flybrain_drive.diagnostic_path_projection import FutureTrace
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=B/'20261029-t1434-body21-flight-cache';m=read(root/'dataset.json');r=read(root/'result.json')
 if sha256(root/'dataset.json')!=r['dataset_manifest_sha256'] or len(m['episodes'])!=8:raise ValueError('source manifest')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),dataset_sha256=sha256(root/'dataset.json'),scope='all eight pre-existing recordings, evaluation-only, no actor truth or fitting',horizons_m=[5,10,20],speed_min_m_s=1,curvature_threshold_per_m=.001))
 summaries=[]
 for ep in m['episodes']:
  recording=Path(ep['original']['recording']);source=read(recording/'result.json');events=recording/'events.jsonl';receipt=recording/'control/api-receipt.json';zipfile=recording/'vision/camera.zip';bundle=read(recording/'vision/bundle.json')
  if source.get('training_admissible') is not False or source.get('expert_labels_available') is not False or sha256(events)!=ep['source_events_sha256'] or sha256(zipfile)!=bundle['sha256']:raise ValueError('source lineage or role changed')
  rows=[json.loads(x) for x in events.read_text(encoding='utf8').splitlines()];trace=FutureTrace(read(receipt)['telemetry']);curves=np.zeros((len(rows),3));valid=np.zeros_like(curves,dtype=bool);speed=np.zeros(len(rows));clock=np.zeros(len(rows))
  with CameraArchive(recording/'vision') as camera:
   for i,e in enumerate(rows):
    if e['index']!=i:raise ValueError('event order')
    raw=camera.read(Path(e['camera']['file']).name)
    if hashlib.sha256(raw).hexdigest()!=e['camera']['file_sha256']:raise ValueError('frame hash')
    frame=json.loads(raw);camera_geometry(frame['views']);front=next(v for v in frame['views'] if v['view_id']=='front')
    start,points,available=trace.at(front['render_started_clock']);forward=np.asarray(front['renderer_only_world_look'])[[0,2]];right=matrix(front['view_matrix_diagnostic'])[:3,0][[0,2]]
    forward=forward/np.linalg.norm(forward);right=right/np.linalg.norm(right);disp=(points-start)[:,[0,2]];f=disp@forward;r=disp@right;squared=(disp**2).sum(axis=1);good=available&(f>0)&(squared>.01)
    curves[i,good]=2*r[good]/squared[good];valid[i]=good;speed[i]=e['native']['fields']['speedKmh']/3.6;clock[i]=e['decision_monotonic_seconds']
  output=run/(ep['file'].replace('.npz','-path.npz'));np.savez_compressed(output,curvature=curves,valid=valid,speed=speed,clock=clock)
  hs={}
  for j,h in enumerate((5,10,20)):
   basic=valid[:,j]&(speed>1);moving=basic&(abs(curves[:,j])>=.001);v=curves[moving,j];left=int((v<0).sum());right=int((v>0).sum())
   hs[str(h)]={'eligible':len(v),'left':left,'right':right,'straight_below_threshold':int((basic & (abs(curves[:,j])<.001)).sum()),'minority_direction':min(left,right),'constant_majority_accuracy':max(left,right)/len(v) if len(v) else None}
  summaries.append({'episode':ep['file'],'role':ep['role'],'original_fit_forbidden':True,'events_sha256':sha256(events),'receipt_sha256':sha256(receipt),'camera_sha256':sha256(zipfile),'path_sha256':sha256(output),'horizons':hs});write_json(run/'progress.json',dict(completed=len(summaries),total=8,episode=ep['file']))
 out={'status':'completed_historical_future_path_coverage','episodes':summaries,'interpretation':'Posthoc diagnostic; future chord is not a steering target or optimal line, correlated samples not independent; no physical exposure synchronization. Original roles do not grant fitting.','fit_allowed':False,'actor_input_future_truth':False,'optimizer_updates':0,'vehicle_control_sent':False}
 write_json(run/'result.json',out);print(json.dumps(out,allow_nan=False))
if __name__=='__main__':main()
