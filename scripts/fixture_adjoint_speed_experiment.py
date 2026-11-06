"""Matched continuation with frozen fixture coefficients and an intact visual adjoint."""
from copy import deepcopy
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.fixture_speed_adjoint import frozen_parameter_speed_command
from scripts.power_bounded_speed_experiment import PowerBoundedSpeedExperiment

class FixtureAdjointSpeedExperiment(PowerBoundedSpeedExperiment):
 def __init__(self,plan,*,fixture_input_adjoint):
  self.fixture_input_adjoint=bool(fixture_input_adjoint);super().__init__(plan,learn_speed=True)
  path=plan['adjoint_parent_checkpoint'];assert sha256(path)==plan['adjoint_parent_checkpoint_sha256'];cp=torch.load(path,map_location='cpu',weights_only=True)
  assert cp['schema']=='flybrain.power_bounded_speed_learning.experimental.v1' and cp['arm']=='speed_policy' and cp['actor_qualified'] is False and cp['maximum_power_input_sensitivity']==plan['maximum_power_input_sensitivity'] and cp['normalization_sha256']==plan['normalization_sha256']
  self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);self.speed_policy.load_state_dict(cp['speed_policy']);self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False);self.speed_origin={k:v.clone() for k,v in self.speed_policy.state_dict().items()}
 def _speed_policy_hook(self,module,args,value):
  h=self.holder
  if self.fixture_input_adjoint and not h.get('speed_policy_off',False) and (h.get('mode')=='synthetic' or h.get('counterfactual_images',False)):
   return frozen_parameter_speed_command(module,*args)
  return super()._speed_policy_hook(module,args,value)
 def make_optimizer(self):
  p=self.port_plan;path=p['adjoint_parent_optimizer'];assert sha256(path)==p['adjoint_parent_optimizer_sha256'];parameters=self.train_parameters();optimizer=torch.optim.Adam([dict(params=[w],lr=lr) for w,lr in zip(parameters,self.learning_rates())]);optimizer.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));assert len(optimizer.param_groups)==4 and [int(optimizer.state[w]['step']) for w in parameters]==p['adjoint_initial_optimizer_steps'] and [g['lr'] for g in optimizer.param_groups]==self.learning_rates();return optimizer
