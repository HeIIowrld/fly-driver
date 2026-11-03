"""Mean-anchored engineering adapter for motion input into the flight visual boundary."""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class CenteredMotionBoundary(nn.Module):
    """Change only motion sensitivity; preserve the source output at train-mean motion.

    This is a learnable approximation, not a measured fly visual receptive field.
    """

    def __init__(self, source, train_mean):
        super().__init__()
        if (not isinstance(source, nn.Linear) or source.in_features != 588
                or not isinstance(train_mean, torch.Tensor) or train_mean.shape != (576,)
                or train_mean.device != source.weight.device
                or train_mean.dtype != source.weight.dtype or not torch.isfinite(train_mean).all()):
            raise ValueError('Original linear boundary and finite 576-feature train mean required')
        self.base = copy.deepcopy(source).requires_grad_(False)
        self.register_buffer('mean', train_mean.detach().clone())
        mask = torch.zeros(4, 9, 16, dtype=source.weight.dtype, device=source.weight.device)
        mask[:, 1:] = 1
        self.register_buffer('motion_mask', mask.flatten())
        self.delta = nn.Parameter(torch.zeros((source.out_features, 576),
                                               dtype=source.weight.dtype,
                                               device=source.weight.device))

    def forward(self, value):
        if value.ndim != 2 or value.shape[1] != 588:
            raise ValueError('Original functional-visual and scene feature contract required')
        centered = (value[:, 12:]-self.mean)*self.motion_mask
        return self.base(value)+F.linear(centered, self.delta)

    @torch.no_grad()
    def materialize(self):
        """Return equivalent original Linear parameters for existing replay consumers."""
        delta = self.delta*self.motion_mask
        weight = self.base.weight.detach().clone()
        weight[:, 12:] += delta
        bias = self.base.bias.detach().clone()-delta@self.mean
        return weight, bias
