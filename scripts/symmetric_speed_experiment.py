"""Continue the latest steering-preserving candidate with symmetric normalized speed input."""
from copy import deepcopy
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand
from flybrain_drive.functional_motion_overlay import exact_model
from scripts.flight_speed_policy_experiment import FlightSpeedPolicyExperiment

class SymmetricSpeedExperiment(FlightSpeedPolicyExperiment):
 def __init__(self,plan,*,learn_speed):
  super().__init__(plan,learn_speed=learn_speed)
  assert sha256(plan['normalized_parent_checkpoint'])==plan['normalized_parent_checkpoint_sha256']
  cp=torch.load(plan['normalized_parent_checkpoint'],map_location='cpu',weights_only=True)
  assert cp['schema']=='flybrain.flight_speed_policy_learning.experimental.v1' and cp['arm']=='neutral_control' and cp['actor_qualified'] is False
  assert all(torch.count_nonzero(v)==0 for v in cp['speed_policy'].values())
  self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port'])
  self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False)
  n=plan['speed_normalization'];self.speed_policy=SymmetricNormalizedFlightSpeedCommand(torch.tensor(n['reference'],dtype=torch.float32),torch.tensor(n['scale'],dtype=torch.float32),max_delta_m_s=plan['speed_max_delta_m_s']);self.speed_policy.requires_grad_(self.learn_speed);self.model.speed_command_boundary=self.speed_policy;self.speed_origin={k:v.clone() for k,v in self.speed_policy.state_dict().items()};self.hooks.append(self.speed_policy.register_forward_hook(self._speed_policy_hook))
 def make_optimizer(self):
  p=self.port_plan;path=p['normalized_parent_optimizer'];assert sha256(path)==p['normalized_parent_optimizer_sha256'];parameters=self.train_parameters();optimizer=torch.optim.Adam([dict(params=[w],lr=lr) for w,lr in zip(parameters[:2],self.learning_rates()[:2])]);optimizer.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));assert [int(optimizer.state[w]['step']) for w in parameters[:2]]==p['initial_optimizer_steps'];assert [g['lr'] for g in optimizer.param_groups]==self.learning_rates()[:2]
  for w,lr in zip(parameters[2:],self.learning_rates()[2:]):optimizer.add_param_group(dict(params=[w],lr=lr))
  return optimizer
 def verify_fixed(self):
  for index,(module,prior) in enumerate(zip([*self.components[:5],self.motion],self.protected)):
   state=module.state_dict()
   if index==0:
    allowed={'speed_command_boundary.projection.weight','speed_command_boundary.projection.bias','speed_command_boundary.reference','speed_command_boundary.scale'};assert allowed.issubset(state);state={k:v for k,v in state.items() if k not in allowed}
    if not exact_model(state,prior):raise ValueError('Original flight graph/legacy sensory components changed')
   elif index==1:
    if not exact_model(state,self.fixed_head):raise ValueError('Fixed signed decoder changed')
   elif index==4:
    if not exact_model({k:v for k,v in state.items() if k!='projection.weight'},self.original_native_buffers):raise ValueError('Native normalization changed')
   elif not exact_model(state,prior):raise ValueError('Frozen sensory component changed')
  state={k:v for k,v in self.calibration.state_dict().items() if k!='motion_projection.weight'};p=self.port_plan;w=self.calibration.projection.weight.detach();native=self.body_port.projection.weight.detach();delta=native-self.native_origin
  if not exact_model(state,self.fixed_overlay) or not torch.equal(self.gain.value,torch.ones_like(self.gain.value)) or not torch.isfinite(w).all() or w.norm()>p['visual_total_norm']+1e-7 or w.abs().max()>2+1e-7 or not torch.isfinite(native).all() or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7:raise ValueError('Frozen parent or cumulative motion/native bounds failed')
  sw,sb=self.speed_policy.projection.weight.detach(),self.speed_policy.projection.bias.detach()
  if self.speed_policy.max_delta_m_s!=p['speed_max_delta_m_s'] or not torch.isfinite(sw).all() or not torch.isfinite(sb).all() or sw.norm()>p['speed_weight_total_norm']+1e-7 or sw.abs().max()>p['speed_weight_absolute_bound']+1e-7 or sb.norm()>p['speed_bias_total_norm']+1e-7 or sb.abs().max()>p['speed_bias_absolute_bound']+1e-7 or sw[0,-1]>0:raise ValueError('Bounded nonpositive direct speed gain required')
  if not self.learn_speed and not exact_model(self.speed_policy.state_dict(),self.speed_origin):raise ValueError('Neutral control policy changed')

  assert torch.equal(self.speed_policy.reference,torch.tensor(p['speed_normalization']['reference'],dtype=torch.float32)) and torch.equal(self.speed_policy.scale,torch.tensor(p['speed_normalization']['scale'],dtype=torch.float32))
