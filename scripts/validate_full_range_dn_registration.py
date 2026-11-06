"""Parent-neutral full graph parity and opened-gate registration gradient diagnostic."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.full_range_dn_registration import FullRangeDNRegistration
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.visual_native_port_experiment import VisualNativePortExperiment
from scripts.sac_steering_experiment import restore

ROOT=Path('runs/visual-native-port-learning-20261104')
class RegistrationProbe(VisualNativePortExperiment):
 def __init__(self,plan):
  super().__init__(plan,train_body=True);cp=torch.load(ROOT/'visual_native-fit-01/ports.pt',map_location='cpu',weights_only=True);self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);self.calibration.requires_grad_(False);self.body_port.requires_grad_(False);self.registration=FullRangeDNRegistration(self.dn.reference,self.dn.scale,maximum=self.dn.maximum)
 def _descending(self,module,args,value):
  static,_,_,_=self.inputs();steer,power=value;current=self.calibration(static,self.dn(static));return steer+self.registration(static,current),power

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=RegistrationProbe(plan);out=begin_run(ROOT/'registration-validation-01',dict(program_sha256=sha256(__file__),module_sha256=sha256('src/flybrain_drive/full_range_dn_registration.py'),parent_checkpoint_sha256=sha256(ROOT/'visual_native-fit-01/ports.pt'),diagnostic_open_gate=.05,seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));sequence=[]
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0)
  for i in range(1024):sequence.append(e.step(i,False).numpy())
  with np.load(ROOT/'visual_native-evaluation-01/predictions.npz') as z:parity=float(abs(np.asarray(sequence)-z['prediction_0'][:1024]).max())
  assert parity<2e-6;task=plan['physical_tasks'][0];begin,end=task['begin'],task['end'];states=torch.load(ROOT/'visual_native-evaluation-01/candidate-prefix-states.pt',map_location='cpu',weights_only=True);prefix=states[0,begin];target=torch.tensor(e.items[0][0]['labels'][begin:end])
  def loss():
   restore(e.model,prefix);p=torch.stack([e.step(i,True) for i in range(begin,end)]);return ((p-target).square()*p.new_tensor([20.,1.,4.])).sum(1).mean()
  value=loss();value.backward();zero_gate_grad=float(e.registration.gate.grad.norm());zero_projection_grad=float(e.registration.projection.weight.grad.norm());assert zero_gate_grad>0 and zero_projection_grad==0
  e.registration.zero_grad(set_to_none=True)
  with torch.no_grad():e.registration.gate.fill_(.05)
  loss().backward();rows=[]
  for name,weight in [('gate',e.registration.gate),('projection',e.registration.projection.weight)]:
   grad=weight.grad.detach().clone();norm=float(grad.norm());assert norm>0;direction=grad/grad.norm();initial=weight.detach().clone();finite=[]
   for step in [.0001,.00002]:
    values=[]
    for sign in [-1,1]:
     with torch.no_grad():weight.copy_(initial+sign*step*direction)
     values.append(float(loss().detach()))
    estimate=(values[1]-values[0])/(2*step);finite.append(dict(step=step,analytic=norm,finite_difference=estimate,relative_error=abs(estimate-norm)/norm))
   with torch.no_grad():weight.copy_(initial)
   rows.append(dict(parameter=name,gradient_norm=norm,finite_differences=finite))
  e.verify_fixed();np.savez_compressed(out/'neutral.npz',prediction=np.asarray(sequence));write_json(out/'result.json',dict(status='completed_full_range_registration_component_validation',neutral_events=1024,neutral_maximum_difference=parity,zero_gate_gradient_norm=zero_gate_grad,zero_gate_projection_gradient_norm=zero_projection_grad,opened_gate_gradients=rows,gradient_validated=all(min(v['relative_error'] for v in r['finite_differences'])<.1 for r in rows),registration_optimizer_updates=0,new_driver_trained=False,actor_qualified=False,vehicle_control_sent=False,scope='Zero-gate parent-preserving component and manually opened .05 gate derivative only. No learned registration candidate or physiological/vehicle-control qualification.'))
 finally:e.close()
if __name__=='__main__':main()
