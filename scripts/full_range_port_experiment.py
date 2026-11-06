"""Full-range visual registration and real-native port learning, fixed flight decoder."""
from copy import deepcopy
import torch
from torch import nn
from flybrain_drive.full_range_dn_registration import FullRangeDNRegistration
from flybrain_drive.functional_motion_overlay import exact_model
from flybrain_drive.sac_unified_student import read
from scripts.visual_native_port_experiment import VisualNativePortExperiment,bounded_step

class RegisteredCurrent(nn.Module):
 def __init__(self,base,initial_gate):
  super().__init__();self.base=deepcopy(base).eval().requires_grad_(False);self.registration=FullRangeDNRegistration(base.reference,base.scale,maximum=base.maximum)
  with torch.no_grad():self.registration.gate.fill_(initial_gate)
 @property
 def projection(self):return self.registration.projection
 @property
 def gate(self):return self.registration.gate
 def forward(self,features,parent_current):return self.registration(features,self.base(features,parent_current))

class FullRangePortExperiment(VisualNativePortExperiment):
 def __init__(self,plan,*,train_body):
  super().__init__(read(plan['base_port_plan']),train_body=train_body);self.port_plan=plan;parent=torch.load(plan['parent_port_checkpoint'],map_location='cpu',weights_only=True);self.calibration.load_state_dict(parent['current_calibration']);self.body_port.load_state_dict(parent['body_port']);self.calibration=RegisteredCurrent(self.calibration,plan['initial_gate']);self.native_origin=self.body_port.projection.weight.detach().clone();self.current_origin=self.calibration.projection.weight.detach().clone();self.bias_origin=self.calibration.projection.bias.detach().clone();self.gate_origin=self.calibration.gate.detach().clone();self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False);self.fixed_current_base={k:v.clone() for k,v in self.calibration.base.state_dict().items()};self.fixed_registration_buffers={k:v.clone() for k,v in self.calibration.registration.state_dict().items() if k not in ['gate','projection.weight','projection.bias']};self.calibration.registration.requires_grad_(True);self.body_port.projection.requires_grad_(self.train_body)
 @property
 def native_parameter_index(self):return 3 if self.train_body else None
 def train_parameters(self):return [self.calibration.projection.weight,self.calibration.projection.bias,self.calibration.gate]+([self.body_port.projection.weight] if self.train_body else [])
 def learning_rates(self):
  p=self.port_plan;return [p['visual_learning_rate'],p['bias_learning_rate'],p['gate_learning_rate']]+([p['native_learning_rate']] if self.train_body else [])
 def project_after_step(self,old):
  p=self.port_plan;bounded_step(self.calibration.projection.weight,old[0],self.current_origin,step_norm=p['visual_step_norm'],total_norm=p['visual_total_norm'],absolute=2.);bounded_step(self.calibration.projection.bias,old[1],self.bias_origin,step_norm=p['bias_step_norm'],total_norm=p['bias_total_norm'],absolute=2.)
  with torch.no_grad():
   gate=self.calibration.gate;delta=gate-old[2];gate.copy_(old[2]+delta*(p['gate_step_norm']/delta.norm().clamp_min(1e-30)).clamp_max(1.));gate.clamp_(p['minimum_gate'],1.)
  if self.train_body and self.body_port.projection.weight.grad is not None:bounded_step(self.body_port.projection.weight,old[3],self.native_origin,step_norm=p['native_step_norm'],total_norm=p['native_total_norm'],entry_delta=p['native_entry_delta'])
 def verify_fixed(self):
  # Parent verifier checks original graph, decoder, native normalization and
  # declared weight/total bounds; registration-specific invariants follow.
  super().verify_fixed()
  if not exact_model(self.calibration.base.state_dict(),self.fixed_current_base):raise ValueError('Frozen parent current codec changed')
  buffers={k:v for k,v in self.calibration.registration.state_dict().items() if k not in ['gate','projection.weight','projection.bias']}
  if not exact_model(buffers,self.fixed_registration_buffers):raise ValueError('Registration normalization changed')
  p=self.port_plan;gate=self.calibration.gate
  if not torch.isfinite(gate).all() or (gate<p['minimum_gate']-1e-7).any() or (gate>1).any() or float((self.calibration.projection.bias-self.bias_origin).norm())>p['bias_total_norm']+1e-7:raise ValueError('Finite projected gate/bias required')
