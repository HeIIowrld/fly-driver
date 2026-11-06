"""Learned vehicle pedal registration from the existing six modeled flight muscles."""
from copy import deepcopy
import torch
from torch import nn

class LearnedFlightPedalReadout(nn.Module):
    """Same muscle inputs/normalization and steering; two supervised pedal rows.

    This is an engineering decoder, not a measured fly circuit or launch rule.
    No speed gate, pedal exclusion, minimum throttle, gear or teacher input.
    """
    def __init__(self,parent):
        super().__init__();self.parent=deepcopy(parent).eval().requires_grad_(False)
        self.pedal_projection=deepcopy(parent.base.projection).requires_grad_(True)
        self.register_buffer('fixed_steering_weight',self.pedal_projection.weight[:1].detach().clone())
        self.register_buffer('fixed_steering_bias',self.pedal_projection.bias[:1].detach().clone())
    def forward(self,muscles):
        original=self.parent(muscles)
        z=(muscles-self.parent.base.training_reference)/self.parent.base.training_scale
        return torch.cat([original[:,:1],self.pedal_projection(z)[:,1:].sigmoid()],1)
    @torch.no_grad()
    def project(self):
        self.pedal_projection.weight.clamp_(-10,10);self.pedal_projection.bias.clamp_(-10,10)
        self.pedal_projection.weight[1,4:].clamp_(.0001,10)
        self.pedal_projection.weight[2,4:].clamp_(-10,-.0001)
        self.pedal_projection.weight[:1].copy_(self.fixed_steering_weight)
        self.pedal_projection.bias[:1].copy_(self.fixed_steering_bias)
    def verify(self):
        w,b=self.pedal_projection.weight,self.pedal_projection.bias
        if not all(torch.isfinite(x).all() for x in [w,b]) or w.abs().max()>10.000001 or b.abs().max()>10.000001 or (w[1,4:]<.0001-1e-8).any() or (w[2,4:]>-.0001+1e-8).any() or not torch.equal(w[:1],self.fixed_steering_weight) or not torch.equal(b[:1],self.fixed_steering_bias):raise ValueError('Signed bounded pedal registration/frozen steering violated')
