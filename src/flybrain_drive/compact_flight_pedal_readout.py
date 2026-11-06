"""Compact balance context for an additive pedal logit adapter."""
import numpy as np
import torch
from torch import nn


def compact_features_np(z, centers, radii):
    q = np.sum(((z[:, None, :4] - centers) / radii[None, :, None])**2, axis=-1)
    return np.maximum(1-q, 0)**3


class CompactFlightPedalReadout(nn.Module):
    def __init__(self, phase_head, centers, radii, coefficients):
        super().__init__()
        n = len(centers)
        if (not 1 <= n <= 48 or centers.shape != (n,4) or radii.shape != (n,)
                or coefficients.shape != (2,n) or any(x.dtype != torch.float32 or not x.isfinite().all()
                    for x in (centers,radii,coefficients)) or (radii <= 0).any()
                or coefficients.abs().max() > 8):
            raise ValueError("Bounded compact balance basis and coefficients required")
        self.phase_head = phase_head
        for name, value in zip(("centers","radii","coefficients"),(centers,radii,coefficients)):
            self.register_buffer(name,value.detach().clone())

    def forward(self, muscles):
        head = self.phase_head
        z = (muscles-head.parent.base.training_reference)/head.parent.base.training_scale
        original = head.parent(muscles)
        base = head.pedal_projection(z)[:,1:] + head.offset(head.features(muscles))
        q = ((z[:,None,:4]-self.centers)/self.radii[None,:,None]).square().sum(-1)
        correction = (1-q).clamp_min(0).pow(3) @ self.coefficients.T
        return torch.cat((original[:,:1], (base+correction).sigmoid()),1)
