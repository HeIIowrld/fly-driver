"""Offline engineering retinal residual -> existing visual-port logit adapter.

No direct vehicle output, changed chemical edge, or biological receptive-field
claim. Zero initialization preserves the original visual port exactly.
"""
import torch
from torch import nn


class RetinalResidualPort(nn.Module):
    def __init__(self, reference, scale, ports, *, maximum_logit_delta=.5):
        super().__init__()
        if (reference.ndim != 1 or len(reference) != 256 or scale.shape != reference.shape
                or reference.dtype != torch.float32 or scale.dtype != reference.dtype
                or reference.device != scale.device or not torch.isfinite(reference).all()
                or not torch.isfinite(scale).all() or torch.any(scale <= 0)
                or type(ports) is not int or ports < 1 or not 0 < maximum_logit_delta <= 1):
            raise ValueError('TRAIN-only calibrated 256 retinal residual features and exact port count required')
        self.register_buffer('reference', reference.detach().clone())
        self.register_buffer('scale', scale.detach().clone())
        self.maximum_logit_delta = float(maximum_logit_delta)
        self.projection = nn.Linear(256, ports, bias=False, device=reference.device)
        nn.init.zeros_(self.projection.weight)

    def forward(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != 256 or len(features) < 1
                or features.device != self.reference.device or features.dtype != self.reference.dtype
                or not torch.isfinite(features).all()):
            raise ValueError('Explicit finite current-camera retinal residual features required')
        value = ((features-self.reference)/self.scale).clamp(-5, 5)
        return self.maximum_logit_delta * torch.tanh(self.projection(value))
