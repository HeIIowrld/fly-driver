"""Fixed full visual routing into existing flight ports, no learned gate escape."""
import torch
from scripts.full_range_port_experiment import FullRangePortExperiment

class FixedFullRoutingExperiment(FullRangePortExperiment):
 def __init__(self,plan,*,train_body):
  if plan['initial_gate']!=1. or plan['minimum_gate']!=1.:raise ValueError('Explicit fixed gate=1 required')
  super().__init__(plan,train_body=train_body);self.calibration.gate.requires_grad_(False)
 @property
 def native_parameter_index(self):return 2 if self.train_body else None
 def train_parameters(self):return [self.calibration.projection.weight,self.calibration.projection.bias]+([self.body_port.projection.weight] if self.train_body else [])
 def learning_rates(self):
  p=self.port_plan;return [p['visual_learning_rate'],p['bias_learning_rate']]+([p['native_learning_rate']] if self.train_body else [])
 def project_after_step(self,old):
  super().project_after_step([old[0],old[1],self.calibration.gate.detach().clone()]+([old[2]] if self.train_body else []))
  if not torch.equal(self.calibration.gate,torch.ones_like(self.calibration.gate)):raise ValueError('Fixed full routing changed')
 def verify_fixed(self):
  super().verify_fixed()
  if self.calibration.gate.requires_grad or not torch.equal(self.calibration.gate,torch.ones_like(self.calibration.gate)):raise ValueError('Gate must remain exactly one and frozen')
