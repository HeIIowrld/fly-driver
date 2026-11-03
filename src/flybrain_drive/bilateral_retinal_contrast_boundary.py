"""Research-only bilateral spatial contrast for the existing flight visual ports.

This is a bounded engineering hypothesis, not a measured fly circuit or driver.
"""
import math
import torch
from torch import nn


class BilateralRetinalContrastBoundary(nn.Module):
    """Project mirror-opponent 4x4 retinal summaries without a fixed scene reference."""

    def __init__(self, port_count, *, normalization_floor=0.02):
        super().__init__()
        if (type(port_count) is not int or port_count <= 0
                or isinstance(normalization_floor, bool)
                or not isinstance(normalization_floor, (float, int))
                or not math.isfinite(normalization_floor) or normalization_floor <= 0):
            raise ValueError('Positive port count and finite normalization floor required')
        self.normalization_floor = float(normalization_floor)
        self.projection = nn.Linear(64, port_count, bias=False)
        nn.init.zeros_(self.projection.weight)

    def features(self, scene, view_valid):
        if (not isinstance(scene, torch.Tensor) or scene.ndim != 2 or scene.shape[1] != 576
                or not torch.isfinite(scene).all() or scene.device != self.projection.weight.device
                or scene.dtype != self.projection.weight.dtype
                or not isinstance(view_valid, torch.Tensor) or view_valid.shape != (len(scene), 4)
                or view_valid.dtype != torch.bool or view_valid.device != scene.device
                or not view_valid.all()):
            raise ValueError('Four finite valid pooled views required')
        image = scene.reshape(-1, 4, 9, 4, 4)[:, :, 0]
        image = image - image.mean(dim=(-2, -1), keepdim=True)
        front = .5 * (image[:, 0] - image[:, 0].flip(-1))
        sides = .5 * (image[:, 1] - image[:, 2].flip(-1))
        rear = .5 * (image[:, 3] - image[:, 3].flip(-1))
        opponent = torch.stack((front, sides, -sides.flip(-1), rear), dim=1)
        rms = (image.square().mean(dim=(1, 2, 3), keepdim=True) + 1e-24).sqrt()
        return (opponent / (rms + self.normalization_floor)).reshape(-1, 64)

    def forward(self, scene, view_valid):
        return .5 * torch.tanh(self.projection(self.features(scene, view_valid)))
