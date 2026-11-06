"""Continue the steering candidate with a direct power-feedback sensitivity limit."""
from copy import deepcopy
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.power_feedback_sensitivity import project_power_sensitivity,power_sensitivity
from scripts.symmetric_speed_experiment import SymmetricSpeedExperiment

class PowerBoundedSpeedExperiment(SymmetricSpeedExperiment):
 def __init__(self,plan,*,learn_speed):
  super().__init__(plan,learn_speed=learn_speed)
  assert sha256(plan['bounded_parent_checkpoint'])==plan['bounded_parent_checkpoint_sha256'];cp=torch.load(plan['bounded_parent_checkpoint'],map_location='cpu',weights_only=True);assert cp['schema']=='flybrain.symmetric_normalized_speed_learning.experimental.v1' and cp['arm']=='neutral_control' and cp['actor_qualified'] is False
  assert torch.count_nonzero(cp['speed_policy']['projection.weight'])==torch.count_nonzero(cp['speed_policy']['projection.bias'])==0
  self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);self.speed_policy.load_state_dict(cp['speed_policy']);self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False);self.speed_origin={k:v.clone() for k,v in self.speed_policy.state_dict().items()}
 def make_optimizer(self):
  p=self.port_plan;path=p['bounded_parent_optimizer'];assert sha256(path)==p['bounded_parent_optimizer_sha256'];parameters=self.train_parameters();optimizer=torch.optim.Adam([dict(params=[w],lr=lr) for w,lr in zip(parameters[:2],self.learning_rates()[:2])]);optimizer.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));assert [int(optimizer.state[w]['step']) for w in parameters[:2]]==p['initial_optimizer_steps'];assert [g['lr'] for g in optimizer.param_groups]==self.learning_rates()[:2]
  for w,lr in zip(parameters[2:],self.learning_rates()[2:]):optimizer.add_param_group(dict(params=[w],lr=lr))
  return optimizer
 def project_after_step(self,old):
  super().project_after_step(old)
  w=self.speed_policy.projection.weight
  if self.learn_speed and w.grad is not None:
   project_power_sensitivity(w,self.speed_policy.scale,max_delta_m_s=self.speed_policy.max_delta_m_s,maximum=self.port_plan['maximum_power_input_sensitivity'])
   assert float((w.detach()-old[2]).norm())<=self.port_plan['speed_weight_step_norm']+1e-7
 def verify_fixed(self):
  super().verify_fixed();assert float(power_sensitivity(self.speed_policy.projection.weight.detach(),self.speed_policy.scale,self.speed_policy.max_delta_m_s))<=self.port_plan['maximum_power_input_sensitivity']+1e-3
