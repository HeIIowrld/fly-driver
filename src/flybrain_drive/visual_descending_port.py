"""Engineering visual-intent input to existing flight DNa04/05 ports.

This is NOT a measured HS/VS->DN synapse or a reconstructed higher brain.
It supplies dimensionless input currents at the existing external command cut.
Vehicle outputs must still traverse the fixed flight graph and wing readout.
"""
import math

import torch
from torch import nn


class VisualDescendingPort(nn.Module):
    def __init__(self, reference, scale, orientations, *, tonic=.001, maximum=.2):
        super().__init__()
        if (reference.shape != (320,) or scale.shape != (320,)
                or reference.dtype != torch.float32 or scale.dtype != reference.dtype
                or not torch.isfinite(reference).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0) or orientations.shape != (4,)
                or not torch.all((orientations == -1) | (orientations == 1))
                or not 0 < tonic < maximum < 1):
            raise ValueError('TRAIN static retinal calibration and original four flight orientations required')
        self.register_buffer('reference', reference.detach().clone())
        self.register_buffer('scale', scale.detach().clone())
        self.register_buffer('orientations', orientations.detach().clone())
        self.register_buffer('tonic_logit', torch.tensor(math.log(tonic/(maximum-tonic)), dtype=reference.dtype))
        self.maximum = float(maximum)
        self.projection = nn.Linear(320, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 320
                or features.device != self.reference.device or features.dtype != self.reference.dtype
                or not torch.isfinite(features).all()):
            raise ValueError('Explicit current-camera static retinal features required')
        value = ((features-self.reference)/self.scale).clamp(-5, 5)
        contrast = 8 * torch.tanh(self.projection(value))
        return self.maximum * (self.tonic_logit + contrast*self.orientations).sigmoid()
