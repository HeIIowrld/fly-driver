"""Same fixed flight graph with separate engineering visual turn/speed intents."""
import torch
from flybrain_drive.visual_flight_speed_intent import VisualFlightSpeedIntent
from scripts.descending_intent_experiment import DescendingExperiment
from flybrain_drive.functional_motion_overlay import exact_model

class TurnSpeedExperiment(DescendingExperiment):
 def __init__(self,plan,*,speed_active=True):
  super().__init__(plan);self.speed_active=bool(speed_active)
  self.speed=VisualFlightSpeedIntent(self.dn.reference,self.dn.scale,self.body_port.reference,self.body_port.scale,max_delta_m_s=plan['maximum_speed_delta_m_s'],maximum_log_gain=plan['maximum_body_log_gain'])
  self.speed.requires_grad_(self.speed_active);self.speed_buffers={k:v.clone() for k,v in self.speed.state_dict().items() if not k.endswith('weight')};self.last_speed=None
  self.hooks.append(self.model.motor.command.register_forward_pre_hook(self._speed_input))
 def _speed_input(self,module,args):
  command,body,valid,goals=args[:4];static,_,native,native_valid=self.inputs()
  if self.holder['mode']=='synthetic':native_valid=torch.zeros_like(native_valid) # no assessed native sample in synthetic fixture
  self.last_speed_native_valid=native_valid.detach()
  target,correction,gain=self.speed.evaluate(static,body[:,16:17],valid[:,16:17],native,native_valid,self.views,goals)
  self.last_speed=dict(target=target.detach(),correction=correction.detach(),gain=gain.detach(),body_speed=body[:,16:17].detach())
  if not self.speed_active:return args
  return (torch.cat([command[:,:1],target],1),*args[1:])
 def train_parameters(self):return [self.dn.projection.weight,self.speed.visual_projection.weight,self.speed.body_gain.weight]
 def verify_fixed(self):
  super().verify_fixed();state=self.speed.state_dict();buffers={k:v for k,v in state.items() if not k.endswith('weight')}
  if not exact_model(buffers,self.speed_buffers):raise ValueError('Speed normalization changed')
  for p in self.speed.parameters():
   if not torch.isfinite(p).all() or float(p.detach().abs().max())>self.plan['maximum_speed_weight_delta']+1e-7:raise ValueError('Speed/gain weight bound exceeded')
   if not self.speed_active and torch.count_nonzero(p):raise ValueError('Disabled speed parameters changed')
 @torch.no_grad()
 def project(self):
  super().project()
  for p in self.speed.parameters():p.clamp_(-self.plan['maximum_speed_weight_delta'],self.plan['maximum_speed_weight_delta'])

def parameter_vector(parameters):return torch.cat([p.detach().reshape(-1) for p in parameters]).numpy().copy()
def gradient_vector(parameters):return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters]).numpy().copy()
