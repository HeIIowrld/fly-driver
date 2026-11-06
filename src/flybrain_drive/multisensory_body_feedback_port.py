"""Experimental native body sensations at existing flight sensory ports.

No direct vehicle output, optimal-temperature lookup, route or teacher input.
Thermal/slip/load interactions are learned vehicle hypotheses, not measured fly
sensory tuning. An absent extension contributes exactly zero.
"""
import torch
from torch import nn

class MultisensoryBodyFeedbackPort(nn.Module):
 def __init__(self,reference,scale,sensory_ports):
  super().__init__()
  if reference.shape!=(19,) or scale.shape!=(19,) or reference.dtype!=torch.float32 or scale.dtype!=reference.dtype or not torch.isfinite(reference).all() or not torch.isfinite(scale).all() or (scale<=0).any() or type(sensory_ports) is not int or sensory_ports<1:raise ValueError('TRAIN-only finite native19 normalization and body-port count required')
  self.register_buffer('reference',reference.detach().clone());self.register_buffer('scale',scale.detach().clone())
  self.projection=nn.Linear(46,sensory_ports,bias=False);nn.init.zeros_(self.projection.weight)
 def forward(self,values,valid):
  if not isinstance(values,torch.Tensor) or values.ndim!=2 or values.shape[1]!=19 or values.dtype!=self.reference.dtype or values.device!=self.reference.device or not isinstance(valid,torch.Tensor) or valid.shape!=values.shape or valid.dtype!=torch.bool or valid.device!=values.device or not torch.isfinite(values).all():raise ValueError('Current finite native19 and explicit per-feature masks required')
  z=((values-self.reference)/self.scale).clamp(-5,5);z=torch.where(valid,z,torch.zeros_like(z))
  slip=z[:,3:7];temperature=z[:,11:15];load=z[:,7:11]
  thermal=torch.where(valid[:,3:7]&valid[:,11:15],slip*temperature,torch.zeros_like(slip))
  traction=torch.where(valid[:,3:7]&valid[:,7:11],slip*load,torch.zeros_like(slip))
  encoded=torch.cat([z,valid.to(values.dtype)-1,thermal,traction],dim=1)
  delta=.25*torch.tanh(self.projection(encoded))
  return torch.where(valid.any(1,keepdim=True),delta,torch.zeros_like(delta))
