"""Engineering muscle-only affine/quadratic steering with unchanged pedals.

Quadratic proxy interactions are virtual vehicle decoding hypotheses, not
measured wing force laws. Explicit float64 decoder arithmetic limits cancellation;
the fixed neural circuit and returned vehicle command remain float32.
"""
import torch
from torch import nn
from .visual_dn_current_calibration import PowerMonotoneVehicleReadout

PAIRS=tuple((i,j) for i in range(6) for j in range(i,6))

class MuscleInteractionVehicleReadout(nn.Module):
 def __init__(self,parent_state,*,degree=1,quadratic_reference=None,quadratic_scale=None):
  super().__init__()
  if type(degree) is not int or degree not in [1,2]:raise ValueError('Explicit affine or quadratic engineering decoder required')
  self.degree=degree;self.base=PowerMonotoneVehicleReadout(parent_state);self.base.eval().requires_grad_(False)
  if (self.base.training_scale<1e-12).any():raise ValueError('Finite assessed muscle normalization required')
  size=21 if degree==2 else 0
  for value in [quadratic_reference,quadratic_scale]:
   if not isinstance(value,torch.Tensor) or value.shape!=(size,) or value.dtype!=torch.float64 or not torch.isfinite(value).all():raise ValueError('Explicit TRAIN-only float64 quadratic normalization required')
  if (quadratic_scale<=0).any():raise ValueError('Positive quadratic scales required')
  self.register_buffer('quadratic_reference',quadratic_reference.detach().clone());self.register_buffer('quadratic_scale',quadratic_scale.detach().clone());self.steering=nn.Linear(6+size,1,dtype=torch.float64)
  with torch.no_grad():self.steering.weight.zero_();self.steering.weight[0,:6].copy_(self.base.projection.weight[0].double());self.steering.bias.copy_(self.base.projection.bias[:1].double())
 def features(self,muscles):
  if not isinstance(muscles,torch.Tensor) or muscles.ndim!=2 or muscles.shape[1]!=6 or muscles.dtype!=torch.float32 or muscles.device!=self.base.training_reference.device or not torch.isfinite(muscles).all() or (muscles<0).any() or (muscles>1).any():raise ValueError('Only finite float32 six bounded original muscle proxies required')
  z=(muscles.double()-self.base.training_reference.double())/self.base.training_scale.double()
  if self.degree==1:return z
  q=torch.stack([z[:,i]*z[:,j] for i,j in PAIRS],1);return torch.cat([z,(q-self.quadratic_reference)/self.quadratic_scale],1)
 def forward(self,muscles):
  features=self.features(muscles)
  if not torch.isfinite(self.steering.weight).all() or not torch.isfinite(self.steering.bias).all():raise ValueError('Finite engineering steering coefficients required')
  steer=self.steering(features).tanh().to(muscles.dtype);return torch.cat([steer,self.base(muscles)[:,1:]],1)
