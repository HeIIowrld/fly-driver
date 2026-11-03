"""Single-frame synthetic retina information probe; diagnostic only, never an actor."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 planpath=B/'20261101-road-signal-decodability-plan/plan.json';plan=read(planpath)
 for path,key in [('src/flybrain_drive/synthetic_road_visual.py','road_source_sha256'),('configs/ac-csp-exterior-front-v3-retina.json','camera_sha256'),('src/flybrain_drive/flyvis_camera.py','projector_sha256')]:
  if sha256(path)!=plan[key]:raise ValueError('declared input changed')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),actor=False,synthetic=True,physical_control=False))
 result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,diagnostic_comparator_only=True)
 try:
  camera=read('configs/ac-csp-exterior-front-v3-retina.json');uv=[(q,r) for q in range(-15,16) for r in range(max(-15,-15-q),min(15,15-q)+1)]
  projector=FlyvisCameraProjector(camera,*np.array(uv).T)
  basis=read('configs/flight-circuit-visual-v6-20261029.json')['scene_basis'];W=FlightScenePool(projector.uv,basis).weight.detach().numpy().astype(np.float64)
  with np.load(B/'20261101-road-scene-pool-audit/pool-audit.npz',allow_pickle=False) as z:pool_error=float(np.max(np.abs(W-z['pool_weight'])))
  if pool_error>1e-7:raise ValueError('pool differs from existing model')
  datasets={}
  for role in ('train','evaluation'):
   rows=[];labels=[];metadata=[]
   for i,seed in enumerate(plan[role+'_seeds']):
    curvature,start=plan[role+'_geometry'][i%len(plan[role+'_geometry'])];road=SyntheticRoadVisual(camera,curvature=curvature,bend_start_m=start)
    for bend in plan['bend_labels']:
     rgb=road.render(bend=bend,forward_m=plan['position_m'],texture_seed=seed)
     retina=np.stack([projector.sample(view) for view in rgb]).astype(np.float64);pooled=np.tanh(retina)@W.T
     rows.append((retina.reshape(-1),pooled.reshape(-1)));labels.append(bend);metadata.append([seed,bend,curvature,start])
    if (i+1)%4==0:write_json(run/'progress.json',dict(role=role,completed=i+1,total=len(plan[role+'_seeds'])))
   datasets[role]=dict(raw=np.stack([r[0] for r in rows]),pooled=np.stack([r[1] for r in rows]),labels=np.array(labels),metadata=np.array(metadata))
  rows=[];arrays={}
  for kind in ('raw','pooled'):
   train=datasets['train'][kind];test=datasets['evaluation'][kind];y=datasets['train']['labels'];yt=datasets['evaluation']['labels'];center=train.mean(0);X=train-center;Z=test-center
   K=X@X.T;ridge=.01*float(np.trace(K)/len(K))
   if ridge<=0:raise ValueError('degenerate features')
   coefficients=np.linalg.solve(K+ridge*np.eye(len(K)),y);w=X.T@coefficients;scores=Z@w;trainscore=X@w
   acc=float(np.mean(np.sign(scores)==yt));paired=float(np.mean(scores[1::2]>scores[::2]));train_acc=float(np.mean(np.sign(trainscore)==y))
   rows.append(dict(feature=kind,feature_count=train.shape[1],train_accuracy=train_acc,evaluation_accuracy=acc,evaluation_paired_ranking=paired,ridge=ridge,score_min=float(scores.min()),score_max=float(scores.max())))
   arrays[kind+'_weights']=w;arrays[kind+'_center']=center;arrays[kind+'_scores']=scores;arrays[kind+'_train_scores']=trainscore
  arrays['train_metadata']=datasets['train']['metadata'];arrays['evaluation_metadata']=datasets['evaluation']['metadata'];output=run/'diagnostic.npz';np.savez_compressed(output,**arrays)
  result.update(status='completed_synthetic_signal_decodability_diagnostic',rows=rows,pool_weight_max_abs_error=pool_error,train_images=len(datasets['train']['labels']),evaluation_images=len(datasets['evaluation']['labels']),validation_used_for_fit=False,outputs_sha256=sha256(output),interpretation='single frame synthetic diagnostic classifiers, no fly physiology, no actor use, no AC track generalization',physical_or_physiology_verified=False)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
