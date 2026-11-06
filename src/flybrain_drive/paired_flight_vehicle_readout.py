"""Engineering vehicle decoder of six fixed flight muscle proxies.

Raw pair reflection flips steer and preserves longitudinal outputs. Increasing
either power proxy cannot decrease gas or increase brake in monotone mode.
These are virtual-body coordinate hypotheses, not measured muscle physiology.
"""
import torch
from torch import nn

class PairedFlightVehicleReadout(nn.Module):
 def __init__(self,difference_scale,context_reference,context_scale,power_reference,power_scale,*,monotone=True,minimum_power_gain=.02,maximum_gain=10.):
  super().__init__()
  for value,shape in [(difference_scale,(3,)),(context_reference,(4,)),(context_scale,(4,)),(power_reference,()),(power_scale,())]:
   if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('Explicit finite TRAIN-only paired muscle normalization required')
  if (difference_scale<=0).any() or (context_scale<=0).any() or power_scale<=0 or type(monotone) is not bool or not 0<minimum_power_gain<maximum_gain<=20:raise ValueError('Positive paired scales and bounded positive power gain required')
  for name,value in [('difference_scale',difference_scale),('context_reference',context_reference),('context_scale',context_scale),('power_reference',power_reference),('power_scale',power_scale)]:self.register_buffer(name,value.detach().clone())
  self.monotone=monotone;self.minimum_power_gain=float(minimum_power_gain);self.maximum_gain=float(maximum_gain)
  self.steering_weight=nn.Parameter(torch.zeros(3));self.context_weight=nn.Parameter(torch.zeros(2,4));self.power_gain=nn.Parameter(torch.full((2,),.1));self.long_bias=nn.Parameter(torch.zeros(2))
 def features(self,muscles):
  if not isinstance(muscles,torch.Tensor) or muscles.ndim!=2 or muscles.shape[1]!=6 or muscles.dtype!=self.difference_scale.dtype or muscles.device!=self.difference_scale.device or not torch.isfinite(muscles).all() or (muscles<0).any() or (muscles>1).any():raise ValueError('Finite bounded six original muscle proxies required')
  difference=muscles[:,1::2]-muscles[:,::2];common=.5*(muscles[:,1::2]+muscles[:,::2]);context=torch.cat([difference[:,:2].abs(),common[:,:2]],1)
  return (difference/self.difference_scale).clamp(-8,8),((context-self.context_reference)/self.context_scale).clamp(-8,8),((common[:,2:3]-self.power_reference)/self.power_scale).clamp(-8,8)
 def forward(self,muscles):
  if any(not torch.isfinite(p).all() or (p.abs()>self.maximum_gain+1e-6).any() for p in self.parameters()) or self.monotone and (self.power_gain<self.minimum_power_gain-1e-7).any():raise ValueError('Finite projected decoder parameters required')
  differential,context,power=self.features(muscles);steer=(differential@self.steering_weight).tanh();gain=torch.stack([self.power_gain[0],-self.power_gain[1]])
  logits=context@self.context_weight.T+power*gain+self.long_bias
  return torch.cat([steer[:,None],logits.sigmoid()],1)
 @torch.no_grad()
 def project(self):
  self.steering_weight.clamp_(-self.maximum_gain,self.maximum_gain);self.context_weight.clamp_(-self.maximum_gain,self.maximum_gain);self.long_bias.clamp_(-self.maximum_gain,self.maximum_gain);self.power_gain.clamp_(self.minimum_power_gain if self.monotone else -self.maximum_gain,self.maximum_gain)
