"""Continuous original flight replay with the new native body port at zero weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder
from scripts.sac_steering_experiment import dataset,assembly
from scripts.train_qualified_sac_unified import context
from scripts.improve_sac_anticipation import command_metrics

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extension',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
 entries,items,extras=dataset();d,c,_=items[2];root=Path(a.extension);er=json.loads((root/'result.json').read_text());contract=json.loads((root/'contract.json').read_text())
 if er['status']!='completed_assessed_native_multisensory_preparation' or sha256(root/'contract.json')!=er['contract_sha256']:raise ValueError('Assessed extension source changed')
 # Construct the live encoder to verify its SDK/evidence/identity contract too.
 encoder=NativeMultisensoryExtensionEncoder(root/'contract.json',native_identity=contract['native_identity'])
 for file in ['extension-2.npz','training-normalization.npz']:
  if sha256(root/file)!=er['files_sha256'][file]:raise ValueError('Native extension sidecar/calibration changed')
 with np.load(root/'extension-2.npz') as z:
  np.testing.assert_array_equal(z['body_packet'],d['body_packet']);np.testing.assert_array_equal(z['body_receipt'],d['body_receipt']);np.testing.assert_array_equal(z['decision_time'],d['decision_time']);values=torch.tensor(z['values']);valid=torch.tensor(z['valid'])
 with np.load(root/'training-normalization.npz') as z:reference=torch.tensor(z['reference']);scale=torch.tensor(z['scale'])
 model,head,dn,base,lineage=assembly(a.source);cp=torch.load(Path(a.source)/'steering.pt',map_location='cpu',weights_only=True);ac=cp['optical']
 optical=RetinalResidualPort(ac['reference'],ac['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(ac)
 port=MultisensoryBodyFeedbackPort(reference,scale,len(model.motor.backbone.body_ports))
 for module in [model,head,dn,optical,port]:module.requires_grad_(False)
 run=begin_run(a.output,dict(arguments=vars(a),seed=780603,program_sha256=sha256(__file__),source_checkpoint_sha256=sha256(Path(a.source)/'steering.pt'),extension_contract_sha256=er['contract_sha256'],entries=entries,lineage=lineage,body_port_indices=model.motor.backbone.body_ports.tolist(),sensory_columns=19,trainable_parameters=0,optimizer_updates=0,scope='Full consumed development continuous replay, zero-neutral body-port integration only; no sensory learning, driving or physiological qualification',actor_deployed=False,vehicle_control_sent=False))
 out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0);holder={'i':None};max_delta=0.;start=time.monotonic()
 def dn_hook(m,args,value):
  steering,power=value;i=holder['i'];return steering+dn(extra[i:i+1]),power
 def optical_hook(m,args,value):
  i=holder['i'];return value+optical(extra[i:i+1,64:])
 def body_hook(m,args,value):
  nonlocal max_delta
  i=holder['i'];expected=torch.tensor(d['body'][i:i+1],dtype=torch.float32)
  if not torch.equal(args[0],expected):raise ValueError('Extra sensations attached to a different current body event')
  delta=port(values[i:i+1],valid[i:i+1]);max_delta=max(max_delta,float(delta.abs().max()));return value+delta
 h=model.motor.command.register_forward_hook(dn_hook);oh=model.visual.register_forward_hook(optical_hook);bh=model.body_feedback.register_forward_hook(body_hook)
 scene=torch.tensor(d['scene'],dtype=torch.float32);extra=torch.tensor(extras[2],dtype=torch.float32);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
 try:
  model.reset_episode();preds=[]
  for i in range(len(scene)):
   holder['i']=i;_,mu=model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=goals,scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i));preds.append(head(mu)[0].numpy().copy())
   if i%1000==0:write_json(run/'progress.json',dict(event=i,total=len(scene),elapsed_seconds=time.monotonic()-start))
  pred=np.asarray(preds)
  with np.load(Path(a.source)/'predictions.npz') as z:error=float(abs(pred-z['dev']).max())
  if max_delta!=0 or error>2e-6:raise ValueError('Zero extension changed source behavior')
  for name,value in model.state_dict().items():
   original=base['model'][name]
   if value.layout!=torch.strided:
    x,y=value.to_sparse_coo().coalesce(),original.to_sparse_coo().coalesce();same=torch.equal(x.indices(),y.indices()) and torch.equal(x.values(),y.values())
   else:same=torch.equal(value,original)
   if not same:raise ValueError('Protected source model changed '+name)
  np.savez_compressed(run/'predictions.npz',prediction=pred)
  torch.save(dict(schema='flybrain.native_multisensory_port.v1',body_port=port.state_dict(),source_checkpoint_sha256=sha256(Path(a.source)/'steering.pt'),extension_contract_sha256=er['contract_sha256'],trained=False,optimizer_updates=0,actor_qualified=False),run/'body-extension.pt')
  out.update(status='completed_zero_neutral_native_body_port_replay',events=len(pred),original_model_state_entries_exact=51,new_port_max_abs_delta=max_delta,source_prediction_max_error=error,extension_valid_counts=valid.sum(0).tolist(),metrics=command_metrics(pred,d['labels'],d['fit_mask']),predictions_sha256=sha256(run/'predictions.npz'),checkpoint_sha256=sha256(run/'body-extension.pt'),trained=False,actor_qualified=False,termination_reason='Entire original episode replayed with exact current same-packet native sensations; zero initialization preserves prior behavior and all old source state')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit integration replay failure')
 finally:h.remove();oh.remove();bh.remove()
 out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
