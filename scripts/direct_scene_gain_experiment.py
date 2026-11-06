"""Continue exact current actor/Adam, optionally learning only a new visual gain."""
from copy import deepcopy
import torch
from flybrain_drive.direct_scene_port_gain import DirectScenePortGain
from flybrain_drive.graph import sha256
from scripts.fixture_adjoint_speed_experiment import FixtureAdjointSpeedExperiment
from scripts.visual_native_port_experiment import bounded_step

class DirectSceneGainExperiment(FixtureAdjointSpeedExperiment):
 def __init__(self,plan,*,learn_direct):
  super().__init__(plan,fixture_input_adjoint=True);self.learn_direct=bool(learn_direct)
  assert sha256(plan['direct_parent_checkpoint'])==plan['direct_parent_checkpoint_sha256'];cp=torch.load(plan['direct_parent_checkpoint'],map_location='cpu',weights_only=True);assert cp['schema']=='flybrain.fixture_adjoint_speed_learning.experimental.v1' and cp['arm']=='input_adjoint'
  self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);self.speed_policy.load_state_dict(cp['speed_policy']);self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False)
  reference=torch.tensor(plan['direct_gain_reference'],dtype=torch.float32);self.direct_gain=DirectScenePortGain(reference);self.direct_gain.requires_grad_(self.learn_direct);self.hooks.append(self.model.visual.register_forward_hook(self._direct_visual,prepend=True))
 def _direct_visual(self,module,args,value):return value if self.holder.get('direct_gain_off',False) else self.direct_gain(value)
 def train_parameters(self):return super().train_parameters()+[self.direct_gain.delta]
 def learning_rates(self):return super().learning_rates()+[self.port_plan['direct_gain_learning_rate']]
 def make_optimizer(self):
  p=self.port_plan;assert sha256(p['direct_parent_optimizer'])==p['direct_parent_optimizer_sha256'];parameters=self.train_parameters();optimizer=torch.optim.Adam([dict(params=[w],lr=lr) for w,lr in zip(parameters[:4],self.learning_rates()[:4])]);optimizer.load_state_dict(torch.load(p['direct_parent_optimizer'],map_location='cpu',weights_only=True));assert [int(optimizer.state[w]['step']) for w in parameters[:4]]==p['direct_initial_optimizer_steps'];optimizer.add_param_group(dict(params=[parameters[4]],lr=self.learning_rates()[4]));return optimizer
 def project_after_step(self,old):
  super().project_after_step(old[:4]);w=self.direct_gain.delta
  if w.grad is not None:bounded_step(w,old[4],torch.zeros_like(w),step_norm=self.port_plan['direct_gain_step_norm'],total_norm=.5,absolute=.25)
 def verify_fixed(self):
  super().verify_fixed();p=self.port_plan;w=self.direct_gain.delta.detach()
  if not torch.equal(self.direct_gain.reference,torch.tensor(p['direct_gain_reference'],dtype=torch.float32)) or not torch.isfinite(w).all() or w.norm()>.5+1e-7 or w.abs().max()>.25+1e-7 or not self.learn_direct and torch.count_nonzero(w):raise ValueError('TRAIN reference and bounded direct visual gain required')
