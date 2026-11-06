"""Learn spatial T4/T5 current routing and optional actual native sensing only."""
from copy import deepcopy
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from flybrain_drive.spatial_motion_dn_overlay import SpatialMotionDNOverlay
from flybrain_drive.functional_motion_overlay import exact_model
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment
from scripts.visual_native_port_experiment import VisualNativePortExperiment,bounded_step

class MotionDNCurrent(SpatialMotionDNOverlay):
 @property
 def projection(self):return self.motion_projection
 @property
 def gate(self):return self.parent.registration.gate

class SpatialMotionDNExperiment(FixedFullRoutingExperiment):
 def __init__(self,plan,*,train_body):
  super().__init__(plan,train_body=train_body)
  if sha256(plan['motion_parent_checkpoint'])!=plan['motion_parent_checkpoint_sha256']:raise ValueError('Exact fixed full-routing parent required')
  cp=torch.load(plan['motion_parent_checkpoint'],map_location='cpu',weights_only=True)
  if cp['schema']!='flybrain.fixed_full_routing_ports.experimental.v1' or cp['actor_qualified'] is not False:raise ValueError('Explicit unqualified source checkpoint required')
  self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);parent=FixedFullRoutingCurrent(self.calibration.base,1.);parent.load_state_dict(self.calibration.state_dict());self.calibration=MotionDNCurrent(parent,self.motion.reference,self.motion.scale);self.current_origin=self.calibration.projection.weight.detach().clone();self.native_origin=self.body_port.projection.weight.detach().clone();self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False);self.fixed_overlay={k:v.clone() for k,v in self.calibration.state_dict().items() if k!='motion_projection.weight'};self.body_port.projection.requires_grad_(train_body)
 @property
 def native_parameter_index(self):return 1 if self.train_body else None
 def train_parameters(self):return [self.calibration.projection.weight]+([self.body_port.projection.weight] if self.train_body else [])
 def learning_rates(self):return [self.port_plan['visual_learning_rate']]+([self.port_plan['native_learning_rate']] if self.train_body else [])
 def _descending(self,module,args,value):
  static,fine,_,_=self.inputs();steer,power=value;port=self.baseline_calibration if self.holder.get('visual_change_off',False) else self.calibration;return steer+port(static,fine,self.dn(static)),power
 def project_after_step(self,old):
  p=self.port_plan;bounded_step(self.calibration.projection.weight,old[0],self.current_origin,step_norm=p['visual_step_norm'],total_norm=p['visual_total_norm'],absolute=2.)
  if self.train_body and self.body_port.projection.weight.grad is not None:bounded_step(self.body_port.projection.weight,old[1],self.native_origin,step_norm=p['native_step_norm'],total_norm=p['native_total_norm'],entry_delta=p['native_entry_delta'])
 def verify_fixed(self):
  VisualNativePortExperiment.verify_fixed(self)
  state={k:v for k,v in self.calibration.state_dict().items() if k!='motion_projection.weight'}
  if not exact_model(state,self.fixed_overlay) or any(p.requires_grad for p in self.calibration.parent.parameters()) or self.calibration.projection.weight.detach().abs().max()>2+1e-7:raise ValueError('Frozen source and finite bounded motion projection required')
