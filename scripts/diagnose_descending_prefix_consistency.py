"""Compare fixed-parent training prefixes with candidate continuous TRAIN trajectories."""
import json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.sac_unified_student import read
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.descending_intent_recalibration import load_descending_intent
from scripts.descending_intent_experiment import DescendingExperiment

def main():
 torch.set_num_threads(1);root=Path('runs/descending-visual-intent-20261104');e=DescendingExperiment(read(root/'plan-01.json'));out=begin_run(root/'prefix-consistency-01',dict(program_sha256=sha256(__file__),scope=__doc__,actor_qualified=False,vehicle_control_sent=False));initial={k:v.clone() for k,v in e.dn.state_dict().items()};reports=[];saved={};start=time.perf_counter()
 try:
  for arm in ('parent','real_only'):
   e.dn.load_state_dict(initial,strict=True)
   if arm!='parent':load_descending_intent(root/f'{arm}-fit-01/descending-intent.pt',e.components,e.motion,e.parent)
   full={}
   for episode in (0,1):
    if arm=='parent':full[episode]=e.anchors[episode]
    else:
     source=root/f'actual-{arm}-{episode}-01';rr=read(source/'result.json');assert sha256(source/'predictions.npz')==rr['predictions_sha256']
     with np.load(source/'predictions.npz') as z:full[episode]=z['prediction'].copy()
   predictions=[];continuous=[];labels=[];fit=[];pre=[]
   for j,task in enumerate(e.real_tasks):
    _,prediction=e.real(task);episode=task['episode'];begin,end=task['begin'],task['end'];d=e.items[episode][0];predictions.append(prediction.detach().numpy());continuous.append(full[episode][begin:end]);labels.append(d['labels'][begin:end]);fit.append(d['fit_mask'][begin:end]);pre.append(e.pre[episode][begin:end]);write_json(out/'progress.json',dict(arm=arm,windows=j+1,total=len(e.real_tasks),elapsed_seconds=time.perf_counter()-start))
   prediction=np.concatenate(predictions);full_prediction=np.concatenate(continuous);target=np.concatenate(labels);fit_mask=np.concatenate(fit);pre_mask=np.concatenate(pre);difference=float(abs(prediction-full_prediction).max());metrics={}
   for role,mask in [('all_selected_fit',fit_mask),('pre',pre_mask)]:metrics[role]=dict(points=int(mask.sum()),fixed_parent_prefix_rmse=np.sqrt(np.mean((prediction[mask].astype(float)-target[mask])**2,axis=0)).tolist(),continuous_candidate_rmse=np.sqrt(np.mean((full_prediction[mask].astype(float)-target[mask])**2,axis=0)).tolist(),max_abs_prediction_difference=float(abs(prediction[mask]-full_prediction[mask]).max()))
   if arm=='parent':assert difference<2e-6
   reports.append(dict(model=arm,selected_windows=93,events=len(prediction),maximum_prediction_difference=difference,metrics=metrics));saved[arm+'_fixed_prefix']=prediction;saved[arm+'_continuous']=full_prediction
  np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_training_prefix_consistency_diagnostic',reports=reports,predictions_sha256=sha256(out/'predictions.npz'),optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,interpretation='Fixed candidate weights; same selected TRAIN windows/inputs but original-parent versus continuously evolving candidate numerical state. Discrepancy is not by itself proof of the cause of physical driving error or gradient direction conflict.'))
 finally:e.close()
if __name__=='__main__':main()
