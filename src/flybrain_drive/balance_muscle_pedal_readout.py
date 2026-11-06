"""Locally learned balance-muscle pedal registration; original power monotonicity."""
from copy import deepcopy
import torch
from torch import nn

class BalanceMusclePedalReadout(nn.Module):
    def __init__(self,parent,centers,width):
        super().__init__();self.parent=deepcopy(parent).eval().requires_grad_(False)
        if centers.ndim!=2 or centers.shape[1]!=4 or not 1<=len(centers)<=24 or not torch.isfinite(centers).all() or width.shape!=() or not torch.isfinite(width) or width<=0:raise ValueError('TRAIN-only finite four-balance-muscle basis required')
        self.register_buffer('centers',centers.clone());self.register_buffer('width',width.clone());self.offset=nn.Linear(len(centers),2,bias=False);nn.init.zeros_(self.offset.weight)
    def features(self,muscles):
        self.parent.features(muscles) # Original bounded six-muscle contract.
        z=(muscles-self.parent.base.training_reference)/self.parent.base.training_scale
        return torch.exp(-.5*(z[:,:4,None].transpose(1,2)-self.centers[None]).square().sum(-1)/self.width.square())
    def forward(self,muscles):
        old=self.parent(muscles);z=(muscles-self.parent.base.training_reference)/self.parent.base.training_scale
        logits=self.parent.base.projection(z)[:,1:]+self.offset(self.features(muscles))
        return torch.cat([old[:,:1],logits.sigmoid()],1)
    def verify(self):
        if not torch.isfinite(self.offset.weight).all() or self.offset.weight.abs().max()>10.000001:raise ValueError('Learned coefficient bounds violated')
