"""Independent checkpoint/fixed-state and saved heldout-metric verification.

No fit, simulator, actuator, new model selection or live actor evaluation.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import assemble_student,MuscleVehicleReadout
from scripts.train_qualified_sac_unified import load_episode

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def same(a,b):
 if not isinstance(a,torch.Tensor):return a==b
 if a.shape!=b.shape or a.dtype!=b.dtype:return False
 if a.layout==torch.strided and b.layout==torch.strided:return torch.equal(a.cpu(),b.cpu())
 a=a.to_sparse_coo().coalesce().cpu();b=b.to_sparse_coo().coalesce().cpu()
 return torch.equal(a.indices(),b.indices()) and torch.equal(a.values(),b.values())
def finite(x):
 if not isinstance(x,torch.Tensor):return True
 return bool(torch.isfinite(x if x.layout==torch.strided else x.to_sparse_coo().coalesce().values()).all())
def metrics(pred,target,mask):
 error=np.asarray(pred,dtype=np.float64)[mask]-target[mask]
 return dict(mae=np.mean(abs(error),0).tolist(),rmse=np.sqrt(np.mean(error**2,0)).tolist(),
  weighted_mse=float(np.mean(error**2*np.array([4.,1.,1.]))))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(1);root=Path(a.fit);result=read(root/'result.json');manifest=read(root/'manifest.json')['config']
 run=begin_run(a.output,dict(arguments=vars(a),seed=780213,program_sha256=sha256(__file__),fit_result_sha256=sha256(root/'result.json'),fit_manifest_sha256=sha256(root/'manifest.json')))
 audit=dict(status='failed',optimizer_updates_in_audit=0,actor_deployed=False,actuator_messages_sent=0)
 try:
  checkpoint=root/'unified-sac-student.pt'
  if (result['status']!='completed_qualified_teacher_distillation' or result['student_optimizer_updates']<=0
      or sha256(checkpoint)!=result['checkpoint_sha256'] or sha256(root/'heldout-predictions.npz')!=result['heldout_predictions_sha256']):raise ValueError('Original completed fit/checkpoint hashes differ')
  split=Path('runs/teacher-synchronized-20261102/student-split-01.json')
  if sha256(split)!=manifest['sealed_split_sha256']:raise ValueError('Prefit split changed')
  model,lineage=assemble_student();state=model.state_dict();trainable={n for n,p in model.named_parameters() if p.requires_grad}
  payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
  if (payload['schema']!='flybrain.qualified_sac_unified_student.v1' or payload['lineage']!=lineage
      or payload['source_plan_sha256']!=sha256(root/'manifest.json') or payload['actor_qualified'] is not False
      or payload['optimizer_updates']!=result['student_optimizer_updates']):raise ValueError('Checkpoint provenance or qualification differs')
  if state.keys()!=payload['model'].keys():raise ValueError('Original unified circuit state keys differ')
  fixed=[k for k in state if k not in trainable];changed=[]
  for k,v in state.items():
   if not finite(payload['model'][k]):raise ValueError('Nonfinite saved neural state')
   if not same(v,payload['model'][k]):
    if k in fixed:raise ValueError('Fixed original circuit/boundary/buffer changed: '+k)
    changed.append(k)
  model.load_state_dict(payload['model'],strict=True);model.motor.backbone.assert_body_boundary_protected()
  _,muscles=model.motor.backbone._read_state_tensor(model.motor.initial_state(1));readout=MuscleVehicleReadout(muscles.shape[-1])
  readout.load_state_dict(payload['readout'],strict=True)
  if not changed or not all(finite(x) for x in payload['readout'].values()):raise ValueError('No sensory learning or invalid readout')
  arguments=manifest['arguments'];train=[load_episode(p) for p in arguments['train']];heldout=[load_episode(p) for p in arguments['heldout']]
  constant=np.concatenate([d['labels'][d['fit_mask']] for d,_,_ in train]).mean(0)
  verified={};effects={}
  with np.load(root/'heldout-predictions.npz',allow_pickle=False) as predictions:
   if set(predictions.files)!={f'{name}_{i}' for name in ('full','vision','body_feedback') for i in range(len(heldout))}:raise ValueError('Saved heldout intervention keys differ')
   for i,(d,c,_) in enumerate(heldout):
    mask=d['fit_mask'];target=d['labels'];rows={}
    for name in ('full','vision','body_feedback'):
     pred=predictions[f'{name}_{i}']
     if (pred.shape!=target.shape or not np.isfinite(pred).all() or np.any(abs(pred[:,0])>1) or np.any(pred[:,1:]<0) or np.any(pred[:,1:]>1)):raise ValueError('Saved three-axis predictions violate contract')
     observed=metrics(pred,target,mask);expected=result['after_fit_heldout'][name][i]
     if any(not np.allclose(observed[k],expected[k],rtol=1e-5,atol=1e-7) for k in observed):raise ValueError('Independent saved heldout metric differs')
     rows[name]=observed
     if name!='full':effects[f'{i}_{name}']=np.mean(abs(pred[mask]-predictions[f'full_{i}'][mask]),0).tolist()
    rows['train_mean_constant']=metrics(np.broadcast_to(constant,target.shape),target,mask)
    initial=metrics(np.broadcast_to([0.,.5,.5],target.shape),target,mask)
    if any(not np.allclose(initial[k],result['before_fit_heldout'][i][k],rtol=1e-5,atol=1e-7) for k in initial):raise ValueError('Original zero-projection before-fit metric differs')
    rows['before_fit']=initial;verified[c['episode_id']]=rows
  audit.update(status='completed_independent_student_fit_audit',checkpoint_sha256=sha256(checkpoint),
   actual_fit_optimizer_updates=result['student_optimizer_updates'],fixed_model_state_entries_verified=len(fixed),
   changed_sensory_parameters=changed,all_nontrainable_state_unchanged=True,protected_haltere_boundary_unchanged=True,
   heldout_metrics=verified,intervention_prediction_mean_absolute_changes=effects,
   interpretation='Independent fixed-state/finite weights/source/array and saved prediction metric checks; no autonomous student lap, physiological validation or cross-car generalization claim')
 except Exception as error:audit.update(error=type(error).__name__,reason=str(error))
 write_json(run/'result.json',audit);print(json.dumps(audit))
 if audit['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
