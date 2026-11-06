"""Engineering full-range registration at four existing flight input ports.

Zero gates preserve the parent exactly. A gated bounded visual branch can open
an inactive population without changing measured chemical topology. This is not
measured DN tuning, a physiological tonic, or an already trained driver.
"""
import torch
from torch import nn

class FullRangeDNRegistration(nn.Module):
 def __init__(self,reference,scale,*,maximum=.2,weight_bound=2.):
  super().__init__()
  if reference.shape!=(320,) or scale.shape!=(320,) or reference.dtype!=torch.float32 or scale.dtype!=reference.dtype or not torch.isfinite(reference).all() or not torch.isfinite(scale).all() or (scale<=0).any() or not 0<maximum<1 or not 0<weight_bound<=4:raise ValueError('Original finite retinal normalization and explicit current bounds required')
  self.register_buffer('reference',reference.detach().clone());self.register_buffer('scale',scale.detach().clone());self.maximum=float(maximum);self.weight_bound=float(weight_bound);self.projection=nn.Linear(320,4);nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias);self.gate=nn.Parameter(torch.zeros(4))
 def forward(self,features,parent_current):
  if features.ndim!=2 or features.shape[1]!=320 or parent_current.shape!=(len(features),4) or features.dtype!=self.reference.dtype or parent_current.dtype!=features.dtype or features.device!=self.reference.device or parent_current.device!=features.device or not torch.isfinite(features).all() or not torch.isfinite(parent_current).all() or (parent_current<0).any() or (parent_current>self.maximum).any():raise ValueError('Original finite image features and bounded parent currents required')
  if not torch.isfinite(self.gate).all() or (self.gate<0).any() or (self.gate>1).any() or any(not torch.isfinite(p).all() or (p.abs()>self.weight_bound+1e-7).any() for p in self.projection.parameters()):raise ValueError('Projected finite registration parameters required')
  z=((features-self.reference)/self.scale).clamp(-5,5);target=self.maximum*self.projection(z).sigmoid();return (parent_current+self.gate*(target-parent_current)).clamp(0,self.maximum)
 @torch.no_grad()
 def project(self):
  self.gate.clamp_(0,1)
  for p in self.projection.parameters():p.clamp_(-self.weight_bound,self.weight_bound)
