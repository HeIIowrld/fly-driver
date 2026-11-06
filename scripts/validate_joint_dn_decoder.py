"""Original-output neutral parity and neural current-calibration finite difference."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from scripts.sac_steering_experiment import restore
from scripts.continuous_descending_experiment import detach_exact

def main():
 torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');e=JointDNDecoderExperiment(plan,4);out=begin_run(root/'validation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(root/'fit-plan-01.json'),actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0));prediction=[]
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0)
  for i in range(1024):
   prediction.append(e.step(i,False).numpy().copy())
   if (i+1)%32==0:detach_exact(e.model)
  parity=float(abs(np.asarray(prediction)-e.anchors[0][:1024]).max());assert parity<2e-6
  warm=torch.load(root/'warm-fit-01/warm-head.pt',map_location='cpu',weights_only=True);e.head.load_state_dict(warm['readout']);task=read(Path(plan['yaw_probes'])/'result.json')['cases'][0]['task'];episode,begin,end=task['episode'],task['begin'],task['end'];e.holder.update(mode='real',episode=episode);weight=e.calibration.projection.weight;initial=weight.detach().clone();target=torch.tensor(e.items[episode][0]['labels'][begin:end],dtype=torch.float32)
  def loss():
   restore(e.model,e.states[episode,begin]);pred=torch.stack([e.step(i,True) for i in range(begin,end)]);return ((pred-target).square()*torch.tensor([20.,1.,4.])).sum(1).mean()
  value=loss();value.backward();gradient=weight.grad.detach().clone();norm=float(gradient.norm());assert norm>0;direction=gradient/gradient.norm();finite=[]
  for step in [.0001,.00002]:
   values=[]
   for sign in [-1,1]:
    with torch.no_grad():weight.copy_(initial+sign*step*direction)
    values.append(float(loss().detach()))
   derivative=(values[1]-values[0])/(2*step);finite.append(dict(step=step,analytic=norm,finite_difference=derivative,relative_error=abs(derivative-norm)/norm))
  with torch.no_grad():weight.copy_(initial)
  e.verify_fixed();np.savez_compressed(out/'parity.npz',prediction=np.asarray(prediction));write_json(out/'result.json',dict(status='completed_joint_dn_neutral_and_gradient_validation',neutral_events=1024,neutral_maximum_difference=parity,neural_current_gradient_norm=norm,finite_differences=finite,gradient_validated=min(v['relative_error'] for v in finite)<.1,parity_sha256=sha256(out/'parity.npz'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Counterfactual parent-prefix gradient diagnostic only; candidate TRAIN fit uses its own uninterrupted numerical history.'))
 finally:e.close()
if __name__=='__main__':main()
