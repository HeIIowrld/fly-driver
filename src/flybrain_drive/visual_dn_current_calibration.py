"""Bounded engineering calibration of four existing flight command currents.

Zero weights preserve the original currents exactly. Rank1 is a scalar visual
axis using the existing port orientations; rank4 learns each port separately.
Neither is a measured synapse or a replacement flight motor circuit.
"""
import torch
from torch import nn

class VisualDNCurrentCalibration(nn.Module):
 def __init__(self,reference,scale,orientations,*,rank=4,maximum=.2,maximum_logit_shift=2.):
  super().__init__()
  if rank not in (1,4) or reference.shape!=(320,) or scale.shape!=(320,) or orientations.shape!=(4,) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in [reference,scale,orientations]) or (scale<=0).any() or not ((orientations==1)|(orientations==-1)).all() or not 0<maximum<1 or not 0<maximum_logit_shift<=4:raise ValueError('Original finite calibration and rank1/rank4 required')
  for name,value in [('reference',reference),('scale',scale),('orientations',orientations)]:self.register_buffer(name,value.detach().clone())
  self.rank=rank;self.maximum=float(maximum);self.maximum_logit_shift=float(maximum_logit_shift);self.projection=nn.Linear(320,rank,bias=False);nn.init.zeros_(self.projection.weight)
 def forward(self,features,current):
  if features.ndim!=2 or features.shape[1]!=320 or current.shape!=(len(features),4) or features.dtype!=self.reference.dtype or current.dtype!=features.dtype or features.device!=self.reference.device or current.device!=features.device or not torch.isfinite(features).all() or not torch.isfinite(current).all() or (current<0).any() or (current>self.maximum).any():raise ValueError('Finite original retinal features and bounded four currents required')
  z=((features-self.reference)/self.scale).clamp(-5,5);response=self.projection(z)
  if self.rank==1:response=response*self.orientations
  shift=self.maximum_logit_shift*response.tanh();base=(current/self.maximum).clamp(1e-6,1-1e-6).logit();delta=self.maximum*((base+shift).sigmoid()-base.sigmoid())
  return (current+delta).clamp(0,self.maximum)

class PowerMonotoneVehicleReadout(nn.Module):
 """Full six-proxy affine decoder; positive scales preserve individual power signs."""
 def __init__(self,state,minimum_power_weight=.0001):
  super().__init__();self.register_buffer('training_reference',state['training_reference'].detach().clone());self.register_buffer('training_scale',state['training_scale'].detach().clone());self.projection=nn.Linear(6,3);self.projection.load_state_dict({k.split('.',1)[1]:v for k,v in state.items() if k.startswith('projection.')});self.minimum_power_weight=float(minimum_power_weight)
  if not torch.isfinite(self.training_reference).all() or not torch.isfinite(self.training_scale).all() or (self.training_scale<=0).any() or not 0<minimum_power_weight<=.1:raise ValueError('Positive original normalization required')
 def forward(self,muscles):
  if muscles.ndim!=2 or muscles.shape[1]!=6 or not torch.isfinite(muscles).all() or (muscles<0).any() or (muscles>1).any():raise ValueError('Bounded modeled muscle proxies required')
  z=self.projection((muscles-self.training_reference)/self.training_scale);return torch.cat([z[:,:1].tanh(),z[:,1:].sigmoid()],1)
 @torch.no_grad()
 def project(self):
  self.projection.weight.clamp_(-10,10);self.projection.bias.clamp_(-10,10);self.projection.weight[1,4:].clamp_(self.minimum_power_weight,10);self.projection.weight[2,4:].clamp_(-10,-self.minimum_power_weight)
