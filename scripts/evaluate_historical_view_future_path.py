"""Evaluate fixed visual-view interventions against an offline future-path diagnostic."""
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
 dataset=B/'20261029-t1434-body21-flight-cache';manifest=read(dataset/'dataset.json');parent=read(dataset/'result.json')
 if sha256(dataset/'dataset.json')!=parent['dataset_manifest_sha256']:raise ValueError('dataset changed')
 ep=next(e for e in manifest['episodes'] if e['file']=='episode-02.npz');root=Path(ep['original']['recording']);result=read(root/'result.json');events=root/'events.jsonl';receipt=root/'control/api-receipt.json'
 if result.get('training_admissible') is not False or result.get('expert_labels_available') is not False or sha256(events)!=ep['source_events_sha256']:raise ValueError('diagnostic-only original source required')
 freeze=B/'20261101-fourview-freeze-historical';fr=read(freeze/'result.json');out=freeze/'episode-02-body-ablation-intents.npz'
 if fr['status']!='completed_historical_fourview_frozen_input_counterfactual' or sha256(out)!=fr['episodes'][0]['output_sha256'] or fr['episodes'][0]['source_events_sha256']!=sha256(events):raise ValueError('original intervention evidence')
 audit=read(root/'vision/bundle.json');archive=root/'vision/camera.zip'
 if sha256(archive)!=audit['sha256']:raise ValueError('camera archive changed')
 rows=[json.loads(x) for x in events.read_text(encoding='utf8').splitlines()];telemetry=read(receipt)['telemetry'];trace=FutureTrace(telemetry)
 with np.load(out,allow_pickle=False) as z:clocks=z['clocks'].copy();intents=z['intents'].copy()
 if intents.shape!=(5,len(rows),2) or len(clocks)!=len(rows):raise ValueError('sequence')
 curves=np.zeros((len(rows),3));mask=np.zeros((len(rows),3),dtype=bool);speed=np.zeros(len(rows))
 with CameraArchive(root/'vision') as camera:
  for i,e in enumerate(rows):
   if e['index']!=i or e['decision_monotonic_seconds']!=clocks[i]:raise ValueError('event clock')
   frame_bytes=camera.read(Path(e['camera']['file']).name)
   if hashlib.sha256(frame_bytes).hexdigest()!=e['camera']['file_sha256']:raise ValueError('frame hash')
   frame=json.loads(frame_bytes);camera_geometry(frame['views']);front=next(v for v in frame['views'] if v['view_id']=='front')
   start,points,available=trace.at(front['render_started_clock']);forward=np.asarray(front['renderer_only_world_look'])[[0,2]];right=matrix(front['view_matrix_diagnostic'])[:3,0][[0,2]]
   forward=forward/np.linalg.norm(forward);right=right/np.linalg.norm(right);displacement=(points-start)[:,[0,2]];f=displacement@forward;v=displacement@right;length=(displacement**2).sum(axis=1);valid=available & (f>0)&(length>.01)
   curves[i,valid]=2*v[valid]/length[valid];mask[i]=valid;speed[i]=e['native']['fields']['speedKmh']/3.6
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(events),receipt_sha256=sha256(receipt),camera_sha256=sha256(archive),intervention_sha256=sha256(out),scope='privileged future route diagnostic only; no actor inputs or fit'))
 reports={}
 for j,distance in enumerate((5,10,20)):
  good=mask[:,j]&(speed>1)&(np.abs(curves[:,j])>=.001); truth=curves[good,j]; signs=-np.sign(truth);values=intents[:,good,0]
  reports[str(distance)]=dict(eligible=int(good.sum()),future_right=int((truth>0).sum()),future_left=int((truth<0).sum()),
      intent_sign_matches=[int((np.sign(v)==signs).sum()) for v in values],
      mean_abs_differences_vs_intact=[float(np.abs(v-values[0]).mean()) if len(v) else None for v in values],
      interpretation='future driven chord direction is not an immediate steering target or optimal racing line')
 target=run/'path-diagnostic.npz';np.savez_compressed(target,decision_clock=clocks,curvature=curves,valid=mask,speed=speed)
 record=dict(status='completed_historical_view_future_path_diagnostic',distances_m=[5,10,20],conditions=['intact','front_frozen','left_frozen','right_frozen','rear_frozen'],reports=reports,targets_sha256=sha256(target),teacher_future_or_pose_actor_input=False,optimizer_updates=0,fit_allowed=False,actor_deployed=False,vehicle_control_sent=False,physical_exposure_sync_verified=False,interpretation='One previously examined driving history, correlated samples; frozen pooled neural features are unrealistic counterfactuals. Future path is privileged evaluation only.')
 write_json(run/'result.json',record);print(json.dumps(record,allow_nan=False))
if __name__=='__main__':main()
