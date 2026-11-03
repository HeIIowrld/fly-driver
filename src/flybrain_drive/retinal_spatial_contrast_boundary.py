"""Experimental per-view retinal spatial contrast to flight visual ports.

This is an engineering hypothesis, not a measured retinal synapse or a driver.
"""
import math

import torch
from torch import nn


class RetinalSpatialContrastBoundary(nn.Module):
    """Center 16 retinal spatial summaries within each of four views."""

    def __init__(self, port_count, *, reference=None, max_contrast_rms=None):
        super().__init__()
        if not isinstance(port_count, int) or port_count <= 0:
            raise ValueError('Positive visual port count required')
        if reference is None:
            reference = torch.zeros(4, 16)
        reference = torch.as_tensor(reference, dtype=torch.float32).detach().clone()
        if reference.shape != (4, 16) or not torch.isfinite(reference).all():
            raise ValueError('Finite four-view retinal reference required')
        reference = reference - reference.mean(dim=-1, keepdim=True)
        self.register_buffer('reference', reference)
        if max_contrast_rms is not None and (not math.isfinite(max_contrast_rms) or max_contrast_rms <= 0):
            raise ValueError('Positive finite contrast limit required')
        self.register_buffer('max_contrast_rms', torch.tensor(0. if max_contrast_rms is None else max_contrast_rms))
        self.projection = nn.Linear(64, port_count, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, scene, view_valid):
        if (not isinstance(scene, torch.Tensor) or scene.ndim != 2
                or scene.shape[1] != 576 or not torch.isfinite(scene).all()
                or not isinstance(view_valid, torch.Tensor)
                or view_valid.shape != (len(scene), 4)
                or view_valid.dtype != torch.bool or not view_valid.all()):
            raise ValueError('Four finite valid pooled views required')
        retina = scene.reshape(-1, 4, 9, 16)[:, :, 0, :]
        centered = retina - retina.mean(dim=-1, keepdim=True) - self.reference
        rms = (centered.square().mean(dim=(1, 2), keepdim=True) + 1e-24).sqrt()
        gain = torch.where(self.max_contrast_rms > 0,
                           (self.max_contrast_rms / rms).clamp(max=1.),
                           torch.ones_like(rms))
        return .5 * torch.tanh(self.projection((centered * gain).reshape(-1, 64)))
