"""Original flight sensory graph with explicit engineering current/readout learning."""
import torch
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration,PowerMonotoneVehicleReadout
from flybrain_drive.functional_motion_overlay import exact_model
from scripts.functional_priority_experiment import Experiment

class JointDNDecoderExperiment(Experiment):
 def __init__(self,plan,rank,warm_state=None):
  super().__init__(plan);self.calibration=VisualDNCurrentCalibration(self.dn.reference,self.dn.scale,self.dn.orientations,rank=rank,maximum=self.dn.maximum,maximum_logit_shift=plan['maximum_logit_shift']);old=self.head.state_dict();self.head=PowerMonotoneVehicleReadout(old,plan['minimum_power_weight'])
  if warm_state is not None:self.head.load_state_dict(warm_state,strict=True)
  self.components=(self.model,self.head,*self.components[2:]);self.gain.requires_grad_(False);self.head.requires_grad_(True);self.yaw_offset=0.;self.speed_offset=0.;self.hooks.append(self.model.motor.command.register_forward_pre_hook(self._command_offset))
 def _command_offset(self,module,args):
  command=args[0]
  if self.yaw_offset==0 and self.speed_offset==0:return args
  return (command+command.new_tensor([[self.yaw_offset,self.speed_offset]]),*args[1:])
 def _descending(self,module,args,value):
  static,_,_,_=self.inputs();steer,power=value;base=self.dn(static)
  if not self.holder.get('calibration_off',False):base=self.calibration(static,base)
  return steer+base,power
 def _visual(self,module,args,value):
  static,fine,_,_=self.inputs();total=value+self.optical(static[:,64:])
  return total if self.holder.get('motion_off',False) else total+self.motion(fine)
 def train_parameters(self):return [self.calibration.projection.weight,*self.head.projection.parameters()]
 @torch.no_grad()
 def project(self):self.calibration.projection.weight.clamp_(-self.plan['maximum_current_weight'],self.plan['maximum_current_weight']);self.head.project()
 def verify_fixed(self):
  for index,(module,old) in enumerate(zip([*self.components[:5],self.motion],self.protected)):
   state=module.state_dict()
   if index==1:
    for key in ['training_reference','training_scale']:
     if not torch.equal(state[key],old[key]):raise ValueError('Original readout normalization changed')
   elif not exact_model(state,old):raise ValueError('Original fixed sensory/chemical component changed '+str(index))
  if not torch.equal(self.gain.value,torch.ones_like(self.gain.value)):raise ValueError('Temporal gain changed')
  if not torch.isfinite(self.calibration.projection.weight).all() or self.calibration.projection.weight.detach().abs().max()>self.plan['maximum_current_weight']+1e-7:raise ValueError('Current calibration bound exceeded')
  w=self.head.projection.weight.detach()
  if (w[1,4:]<self.plan['minimum_power_weight']-1e-8).any() or (w[2,4:]>-self.plan['minimum_power_weight']+1e-8).any():raise ValueError('Individual power sign failed')
