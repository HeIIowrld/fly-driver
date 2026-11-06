"""Bounded engineering recalibration of existing direct scene visual-port logits.

This preserves measured chemical wiring. It is not measured fly receptive-field
adaptation. No body, map, command or teacher enters the module.
"""
import torch
from torch import nn

class DirectScenePortGain(nn.Module):
 def __init__(self,reference):
  super().__init__()
  if not isinstance(reference,torch.Tensor) or reference.ndim!=1 or reference.dtype!=torch.float32 or not torch.isfinite(reference).all() or not len(reference):raise ValueError('Finite TRAIN-only original direct-port reference required')
  self.register_buffer('reference',reference.detach().clone());self.delta=nn.Parameter(torch.zeros_like(reference))
 def forward(self,logits):
  if not isinstance(logits,torch.Tensor) or logits.ndim!=2 or logits.shape[1]!=len(self.reference) or logits.dtype!=self.reference.dtype or logits.device!=self.reference.device or not torch.isfinite(logits).all():raise ValueError('Finite original direct visual-port logits required')
  if not torch.isfinite(self.delta).all() or self.delta.detach().abs().max()>.25+1e-7 or self.delta.detach().norm()>.5+1e-7:raise ValueError('Bounded direct scene coefficients required')
  return logits+.5*torch.tanh(self.delta*(logits-self.reference))
