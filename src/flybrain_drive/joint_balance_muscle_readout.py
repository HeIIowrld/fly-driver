"""Joint learned motor registration from six muscle proxies and balance-only context."""
from copy import deepcopy
import torch
from .balance_muscle_pedal_readout import BalanceMusclePedalReadout
class JointBalanceMusclePedalReadout(BalanceMusclePedalReadout):
    def __init__(self,parent,centers,width):
        super().__init__(parent,centers,width);self.pedal_projection=deepcopy(parent.base.projection)
    def forward(self,muscles):
        old=self.parent(muscles);z=(muscles-self.parent.base.training_reference)/self.parent.base.training_scale
        logits=self.pedal_projection(z)[:,1:]+self.offset(self.features(muscles))
        return torch.cat([old[:,:1],logits.sigmoid()],1)
    def verify(self):
        super().verify();w,b=self.pedal_projection.weight,self.pedal_projection.bias
        if not torch.isfinite(w).all() or not torch.isfinite(b).all() or w.abs().max()>10.000001 or b.abs().max()>10.000001 or (w[1,4:]<.0001-1e-8).any() or (w[2,4:]>-.0001+1e-8).any():raise ValueError('Global signed muscle power registration changed')
