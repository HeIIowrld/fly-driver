"""Zero-preserving modeled T4/T5 spatial residual at four existing flight DN inputs.

Engineering learned input registration, not measured new synapses or a driver.
The original four-view/type/grid structure remains in the input. No body,
world pose, heading, target or vehicle action enters this module.
"""
from copy import deepcopy
import torch
from torch import nn
from .fixed_full_routing_candidate import FixedFullRoutingCurrent
from .motion_spatial_residual_port import MOTION_FEATURES

class SpatialMotionDNOverlay(nn.Module):
 def __init__(self,parent,reference,scale):
  super().__init__()
  if type(parent) is not FixedFullRoutingCurrent or not torch.equal(parent.registration.gate,torch.ones_like(parent.registration.gate)) or reference.shape!=(MOTION_FEATURES,) or scale.shape!=reference.shape or reference.dtype!=torch.float32 or scale.dtype!=reference.dtype or reference.device!=scale.device or not torch.isfinite(reference).all() or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Fixed full-routing parent and original TRAIN motion2048 normalization required')
  self.parent=deepcopy(parent).eval().requires_grad_(False);self.register_buffer('motion_reference',reference.detach().clone());self.register_buffer('motion_scale',scale.detach().clone());self.motion_projection=nn.Linear(MOTION_FEATURES,4,bias=False,device=reference.device);nn.init.zeros_(self.motion_projection.weight)
 def forward(self,static,motion,old_current):
  if not isinstance(motion,torch.Tensor) or motion.ndim!=2 or motion.shape!=(len(static),MOTION_FEATURES) or motion.dtype!=self.motion_reference.dtype or motion.device!=self.motion_reference.device or not torch.isfinite(motion).all() or not torch.equal(self.parent.registration.gate,torch.ones_like(self.parent.registration.gate)):raise ValueError('Explicit finite same-event motion2048 and fixed parent gate required')
  if not torch.isfinite(self.motion_projection.weight).all() or (self.motion_projection.weight.abs()>2+1e-7).any():raise ValueError('Finite projected motion weights required')
  parent=self.parent(static,old_current);reg=self.parent.registration;z=((static-reg.reference)/reg.scale).clamp(-5,5);base_logits=reg.projection(z);fine=((motion-self.motion_reference)/self.motion_scale).clamp(-5,5);target=reg.maximum*(base_logits+self.motion_projection(fine)).sigmoid();base_target=reg.maximum*base_logits.sigmoid()
  # Subtract identical expressions at W=0: preserve the exact parent, including
  # its FP32 rounding history. No bounded logit-shift codec is reintroduced.
  return (parent+(target-base_target)).clamp(0,reg.maximum)
