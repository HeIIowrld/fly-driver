"""Reference-clamp only the learned native calibration, preserving original body feedback."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.symmetric_speed_experiment import SymmetricSpeedExperiment as VisualNativePortExperiment
from scripts.sac_steering_experiment import restore

ROOT=Path('runs/symmetric-speed-learning-20261104')
class FeatureAblationExperiment(VisualNativePortExperiment):
 def _body(self,module,args,value):
  group=self.holder.get('reference_group')
  if group is None or not self.train_body:return super()._body(module,args,value)
  _,_,native,valid=self.inputs()
  if group=='native_change_off':return value+self.baseline_native_port(native,valid)
  modified=native.clone()
  if group=='acceleration_reference':modified[:,:3]=self.body_port.reference[:3]
  elif group=='temperature_reference':modified[:,11:15]=self.body_port.reference[11:15]
  elif group!='identity_compose':raise ValueError('Declared calibration feature ablation required')
  # Algebraically base(actual)+new(modified)-base(modified). This ordering gives
  # an exact zero correction for identity input and leaves old feedback intact.
  correction=self.baseline_native_port(native,valid)-self.baseline_native_port(modified,valid)
  return value+(self.body_port(modified,valid)+correction)

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');tasks=read('runs/multicorner-command-20261104/protocol-01.json')['tasks'];e=FeatureAblationExperiment(plan,learn_speed=True);fit=ROOT/'speed_policy-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);e.speed_policy.requires_grad_(False);evaluation=ROOT/'speed_policy-evaluation-01';er=read(evaluation/'result.json');assert sha256(evaluation/'candidate-prefix-states.pt')==er['prefix_states_sha256'];states=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);saved={};reports=[];out=begin_run(ROOT/'native-feature-ablation-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'ports.pt'),source_prefix_sha256=sha256(evaluation/'candidate-prefix-states.pt'),seed=None,optimizer_updates=0,scope=__doc__+' Native19 learned increment only; original body21 and original native baseline retained. Reference clamping is a model intervention, not altered tyre dynamics or physiological evidence.',actor_qualified=False,vehicle_control_sent=False));maximum_identity=0.
 try:
  with np.load(evaluation/'predictions.npz') as z:
   for j,task in enumerate(tasks):
    episode,begin,end=task['episode'],task['begin'],task['end'];fine=z[f'prediction_{episode}'][begin:end];d=e.items[episode][0];valid=d['fit_mask'][begin:end];row=dict(task=task,groups={})
    for group in ['identity_compose','native_change_off','acceleration_reference','temperature_reference']:
     restore(e.model,states[episode,begin]);e.holder.clear();e.holder.update(mode='real',episode=episode,reference_group=group);p=np.asarray([e.step(i,False).numpy() for i in range(begin,end)]);saved[f'case{j}_{group}_prediction']=p;error=float(abs(p-fine).max());row['groups'][group]=dict(maximum_action_difference=error,mean_action_difference=(p-fine).mean(0).tolist(),steering_rmse=None if not valid.any() else float(np.sqrt(np.mean((p[valid,0]-d['labels'][begin:end][valid,0])**2))))
     if group=='identity_compose':maximum_identity=max(maximum_identity,error)
    reports.append(row)
  assert maximum_identity<2e-6;e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_learned_native_feature_ablation',reports=reports,maximum_identity_difference=maximum_identity,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,actual_tyre_adaptation_verified=False))
 finally:e.close()
if __name__=='__main__':main()
