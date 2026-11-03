"""Distill qualified SAC demonstrations into one recurrent flight circuit.

Train only engineering visual/body boundaries and a three-axis readout of the
existing six flight-muscle activities. Chemical topology, protected haltere
mapping and physiological/provenance labels stay fixed. No live controls.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import assemble_student,read,MuscleVehicleReadout,detach_recurrence
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.racing_goal import racing_goal


def load_episode(path):
 root=Path(path);result=read(root/'result.json');contract=read(root/'contract.json')
 if (result['status']!='completed_qualified_actor_preprocessing' or
     sha256(root/'actor-only.npz')!=result['arrays_sha256'] or
     sha256(root/'contract.json')!=result['contract_sha256']):raise ValueError('Qualified actor data/provenance changed')
 if contract.get('unqualified_pixel_inputs')!=0:raise ValueError('Unqualified pixels may not enter even frozen visual recurrence')
 qm=read(root/'manifest.json')['config'];cert=Path(qm['arguments']['qualification'])/'result.json'
 if (sha256(cert)!=qm['qualification_sha256'] or sha256(cert)!=contract['source_qualification_sha256'] or not read(cert)['training_admissible']):
  raise ValueError('Independent camera/body qualification changed or absent')
 with np.load(root/'actor-only.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 expected={'scene','body','body_valid','view_body','view_body_valid','decision_time','body_receipt',
  'view_receipts','render_lowers','camera_receipt','labels','camera_index','body_packet','view_packets','fit_mask','label_valid','teacher_request_index'}
 if set(data)!=expected:raise ValueError('Only the declared actor arrays and original clock/packet audit indices allowed')
 n=len(data['labels'])
 shapes={'scene':(n,576),'body':(n,21),'body_valid':(n,21),'view_body':(n,4,21),
  'view_body_valid':(n,4,21),'view_receipts':(n,4),'render_lowers':(n,4),'labels':(n,3),'view_packets':(n,4)}
 if any(data[k].shape!=v for k,v in shapes.items()) or n<100:raise ValueError('Complete finite actor episode dimensions required')
 if any(not np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite actor array')
 if data['label_valid'].dtype!=np.bool_ or np.any(data['fit_mask']&~data['label_valid']):raise ValueError('Missing labels must remain explicit and unsupervised')
 if data['fit_mask'].dtype!=np.bool_ or data['fit_mask'].shape!=(n,) or data['fit_mask'].sum()<100:raise ValueError('Qualified unrestricted full-lap supervision mask required')
 if data['body_valid'].dtype!=np.bool_ or data['view_body_valid'].dtype!=np.bool_:raise ValueError('Original boolean sensing masks required')
 if (np.any(abs(data['labels'][:,0])>1.00001) or np.any(data['labels'][:,1:]<0) or np.any(data['labels'][:,1:]>1)):
  raise ValueError('Original three-axis vehicle units outside contract')
 return data,contract,dict(directory=str(root),arrays_sha256=result['arrays_sha256'],contract_sha256=result['contract_sha256'])


def context(data,contract,index,intervention=None):
 t=float(data['decision_time'][index]);receipt=float(data['body_receipt'][index]);episode=contract['episode_id'];h=contract['body_contract_sha256']
 values=data['body'][index].copy();valid=data['body_valid'][index].copy()
 hist=data['view_body'][index].copy();hm=data['view_body_valid'][index].copy()
 if intervention=='body_feedback':
  # Neutralize experimentally mapped tilt/load/tyre/speed/wheel channels and
  # yaw feedback, retaining speed for the exact neutral command cancellation.
  keep=values[16];values[:]=0;values[16]=keep;hist[:]=0
 obs=NavigationBodyObservation(episode,int(data['body_packet'][index]),receipt,t,values,valid,h)
 views=tuple(NavigationBodyObservation(episode,int(data['view_packets'][index,j]),float(data['view_receipts'][index,j]),
  float(data['camera_receipt'][index]),hist[j].copy(),hm[j].copy(),h) for j in range(4))
 return per_view_temporal_body_context(views,obs,
  render_start_lowers=tuple(map(float,data['render_lowers'][index])),
  camera_receipt_time=float(data['camera_receipt'][index]),decision_time=t)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',nargs='+',required=True);p.add_argument('--heldout',nargs='+',required=True)
 p.add_argument('--epochs',type=int,default=4);p.add_argument('--window',type=int,default=8);p.add_argument('--output',required=True);a=p.parse_args()
 plan_path=Path('runs/teacher-synchronized-20261102/student-fit-plan-v2.json');plan=read(plan_path)
 if a.epochs!=plan['fit']['epochs'] or a.window!=plan['fit']['window'] or sha256('src/flybrain_drive/sac_unified_student.py')!=plan['circuit']['assembly_sha256']:
  raise ValueError('Sealed fixed fit schedule/assembly differs')
 torch.set_num_threads(1);torch.manual_seed(780205);np.random.seed(780205)
 if len(a.train)!=2 or len(a.heldout)!=1:raise ValueError('Sealed split requires first2qualified train and third heldout')
 train=[load_episode(p) for p in a.train];heldout=[load_episode(p) for p in a.heldout]
 if set(c['episode_id'] for _,c,_ in train)&set(c['episode_id'] for _,c,_ in heldout):raise ValueError('Independent physically restarted heldout episode required')
 split_path=Path('runs/teacher-synchronized-20261102/student-split-01.json');split=read(split_path)
 actual=[(role,Path(e['directory']).resolve(),c['episode_id'],e['arrays_sha256'],e['contract_sha256'])
  for role,items in [('train',train),('heldout',heldout)] for _,c,e in items]
 sealed=[(e['role'],Path(e['data']).resolve(),e['episode_id'],e['arrays_sha256'],e['contract_sha256']) for e in split['episodes']]
 if (split['student_optimizer_updates_at_sealing']!=0 or split['fit_plan_sha256']!=sha256(plan_path)
     or actual!=sealed or len({row[2] for row in actual})!=3):raise ValueError('Immutable chronological prefit split differs')
 model,lineage=assemble_student();device='cuda' if torch.cuda.is_available() else 'cpu'
 model.motor.backbone.circuit.sparse_backend='cached_csr_gpu' if device=='cuda' else 'cached_csr_cpu'
 model=model.to(device);_,muscles=model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
 readout=MuscleVehicleReadout(muscles.shape[-1]).to(device)
 parameters=[p for p in model.parameters() if p.requires_grad]+list(readout.parameters())
 run=begin_run(a.output,dict(arguments=vars(a),seed=780205,program_sha256=sha256(__file__),
  assembly_sha256=sha256('src/flybrain_drive/sac_unified_student.py'),sealed_plan_sha256=sha256(plan_path),sealed_split_sha256=sha256(split_path),lineage=lineage,
  train_sources=[r for _,_,r in train],sealed_heldout_sources=[r for _,_,r in heldout],
  split='Whole separately restarted episodes; heldout never used for optimizer/early stopping',
  optimizer='Adam lr=.001; fixed epoch count; window detach preserves numerical recurrent values',
  targets='Actual 7M teacher requested steering/.6, throttle, brake; pedals remain separate',
  trainable_parameters=sum(p.numel() for p in parameters),fixed_graph_neurons=model.motor.neurons,
  device=device,actor_deployed=False,exact_exposure_timestamp_verified=False))
 result=dict(status='failed',student_optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False,
  student_autonomous_lap_verified=False,physiological_validation=False)
 optimizer=torch.optim.Adam(parameters,lr=.001);weights=torch.tensor([4.,1.,1.],device=device)
 goals=torch.tensor(racing_goal(300,1),device=device);view=torch.ones((1,4),dtype=torch.bool,device=device)
 fixed_graph=model.motor.backbone.circuit.matrix.detach().to_sparse_coo().coalesce().cpu()
 fixed_graph_values=fixed_graph.values().clone();fixed_graph_indices=fixed_graph.indices().clone()
 initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
 initial_readout=copy.deepcopy(readout.state_dict());updates=0
 def rollout(item,learn=False,intervention=None,calibration=None):
  nonlocal updates
  data,contract,_=item;model.reset_episode();predictions=[];losses=[];window_losses=[]
  supervised=np.flatnonzero(data['fit_mask']);first_fit,last_fit=int(supervised[0]),int(supervised[-1])
  body_hook=model.body_feedback.register_forward_hook(lambda module,args,value:torch.zeros_like(value)) if intervention=='body_feedback' else None
  scene=torch.tensor(data['scene'],dtype=torch.float32,device=device);labels=torch.tensor(data['labels'],dtype=torch.float32,device=device)
  if intervention=='vision':scene=torch.zeros_like(scene)
  for i in range(len(labels)):
   if i%1000==0:write_json(run/'progress.json',dict(phase='training' if learn else 'calibration' if calibration is not None else 'evaluation',
    intervention=intervention,optimizer_updates=updates,episode=contract['episode_id'],model_event=i,total_events=len(data['labels'])))
   ctx=context(data,contract,i,intervention);fn=model.sensory_training_decision if learn and first_fit<=i<=last_fit else model.sensory_decision
   _,muscle=fn(decision_time=float(data['decision_time'][i]),racing_goals=goals,
    scene_features=scene[i:i+1],scene_valid=view,body_context=ctx)
   if calibration is not None and data['fit_mask'][i]:calibration.append(muscle.detach().cpu())
   prediction=readout(muscle);loss=((prediction[0]-labels[i]).square()*weights).mean()
   predictions.append(prediction[0].detach().cpu().numpy());losses.append(float(loss.detach()))
   if learn:
    if data['fit_mask'][i]:window_losses.append(loss)
    if (i+1)%a.window==0 or i==len(labels)-1:
     if window_losses:
      optimizer.zero_grad(set_to_none=True);torch.stack(window_losses).mean().backward()
      torch.nn.utils.clip_grad_norm_(parameters,1.)
      if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Nonfinite gradient')
      optimizer.step();updates+=1;window_losses.clear()
      if updates%50==0:write_json(run/'progress.json',dict(optimizer_updates=updates,decision=i,episode=contract['episode_id']))
     detach_recurrence(model)
  if body_hook is not None:body_hook.remove()
  predictions=np.asarray(predictions);mask=data['fit_mask'];target=data['labels'];error=(predictions-target)[mask]
  return dict(episode=contract['episode_id'],supervised_decisions=int(mask.sum()),model_events=len(mask),unlabelled_or_burnin_model_events=int((~mask).sum()),weighted_mse=float(np.mean(np.asarray(losses)[mask])),
   mae=np.mean(abs(error),0).tolist(),rmse=np.sqrt(np.mean(error**2,0)).tolist()),predictions
 try:
  model.eval();readout.eval()
  with torch.no_grad():
   calibration=[]
   for item in train:rollout(item,calibration=calibration)
   activities=torch.cat(calibration)
   readout.training_reference.copy_(activities.mean(0).to(device))
   readout.training_scale.copy_(activities.std(0).clamp_min(.001).to(device))
   write_json(run/'readout-calibration.json',dict(source='Training episodes only, original unmodified recurrent flight activity',
    raw_training_std=activities.std(0).tolist(),reference=readout.training_reference.cpu().tolist(),scale=readout.training_scale.cpu().tolist(),minimum_engineering_scale=.001))
   before=[rollout(x)[0] for x in heldout]
  logs=[]
  for epoch in range(a.epochs):
   model.train();readout.train();metrics=[rollout(x,True)[0] for x in train]
   logs.append(dict(epoch=epoch+1,train=metrics,optimizer_updates=updates));write_json(run/'epochs.json',logs)
   torch.save(dict(model=model.state_dict(),readout=readout.state_dict(),optimizer=optimizer.state_dict(),
    completed_epoch=epoch+1,optimizer_updates=updates,actor_qualified=False),run/f'epoch-{epoch+1:02d}.pt')
  model.eval();readout.eval();evaluations={};preds={}
  with torch.no_grad():
   for intervention in (None,'vision','body_feedback'):
    name=intervention or 'full';pairs=[rollout(x,False,intervention) for x in heldout]
    evaluations[name]=[x[0] for x in pairs]
    for i,(_,pred) in enumerate(pairs):preds[f'{name}_{i}']=pred
  after=model.motor.backbone.circuit.matrix.detach().to_sparse_coo().coalesce().cpu()
  if not torch.equal(after.indices(),fixed_graph_indices) or not torch.equal(after.values(),fixed_graph_values):raise ValueError('Fixed chemical graph changed')
  model.motor.backbone.assert_body_boundary_protected()
  changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
  if updates==0 or not any(v>0 for v in changes.values()):raise ValueError('No actual sensory adapter learning')
  checkpoint=run/'unified-sac-student.pt';torch.save(dict(schema='flybrain.qualified_sac_unified_student.v1',
   model=model.cpu().state_dict(),readout=readout.cpu().state_dict(),lineage=lineage,actor_qualified=False,
   optimizer_updates=updates,engineering_boundaries=True,source_plan_sha256=sha256(run/'manifest.json')),checkpoint)
  np.savez_compressed(run/'heldout-predictions.npz',**preds)
  constant=np.concatenate([x[0]['labels'][x[0]['fit_mask']] for x in train]).mean(0)
  baseline=[dict(episode=c['episode_id'],mae=np.mean(abs(d['labels'][d['fit_mask']]-constant),0).tolist(),
   rmse=np.sqrt(np.mean((d['labels'][d['fit_mask']]-constant)**2,0)).tolist()) for d,c,_ in heldout]
  result.update(status='completed_qualified_teacher_distillation',student_optimizer_updates=updates,
   checkpoint_sha256=sha256(checkpoint),heldout_predictions_sha256=sha256(run/'heldout-predictions.npz'),
   before_fit_heldout=before,after_fit_heldout=evaluations,train_constant_baseline=baseline,
   adapter_max_absolute_changes=changes,fixed_chemical_graph_unchanged=True,
   protected_haltere_boundary_unchanged=True,recurrent_numeric_resets='Only at whole episode restarts',
   ablation_scope='Information-clamp interventions only; no missing-sensor fallback or physiological lesion claim. Body clamp zeroes learned body feedback and yaw/tilt/load/tyre/wheel signals while current speed is retained solely for exact neutral external command cancellation.',
   interpretation='Experimental offline teacher distillation; heldout imitation and intervention effects, no autonomous student driving or biological tuning claim')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),student_optimizer_updates=updates)
 write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
