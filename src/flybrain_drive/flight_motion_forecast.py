"""Experimental causal one-step motion-vision extrapolation, not a second controller."""
import math

import torch
from torch import nn


class CausalMotionForecast(nn.Module):
    """Shared bounded temporal extrapolation for the eight T4/T5-like channels.

    The current and previous pooled 4-view scenes are the only inputs; this
    research approximation has no measured fly physiology or actuator access.
    """

    def __init__(self):
        super().__init__()
        self.raw_gain = nn.Parameter(torch.zeros(8))

    def forward(self, current, previous, elapsed):
        if (not isinstance(current, torch.Tensor) or current.ndim != 2 or current.shape[1] != 576 or len(current) < 1
                or not torch.is_floating_point(current) or not torch.isfinite(current).all()
                or current.dtype != self.raw_gain.dtype or current.device != self.raw_gain.device):
            raise ValueError('Finite batched four-view pooled scenes on the original device required')
        if previous is None:
            if elapsed is not None:
                raise ValueError('Previous observation missing for an ongoing episode')
            return current
        if (not isinstance(previous, torch.Tensor) or previous.shape != current.shape
                or previous.device != current.device or previous.dtype != current.dtype
                or not torch.isfinite(previous).all() or not isinstance(elapsed, (float, int))
                or isinstance(elapsed, bool) or not math.isfinite(elapsed) or not .036 <= elapsed <= .044):
            raise ValueError('Original preceding scene and 40ms cadence within 4ms required')
        now = current.reshape(-1, 4, 9, 16)
        prior = previous.reshape(-1, 4, 9, 16)
        gains = self.raw_gain.tanh().reshape(1, 1, 8, 1)
        motion = now[:, :, 1:] + gains * (now[:, :, 1:] - prior[:, :, 1:])
        return torch.cat((now[:, :, :1], motion), dim=2).reshape(len(current), 576)
