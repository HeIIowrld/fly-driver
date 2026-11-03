"""Experimental speed command from the existing flight state and body speed.

This stateless engineering adapter is not a measured fly circuit or a driver.
"""
import math

import torch
from torch import nn


class FlightStateSpeedCommand(nn.Module):
    """Predict a bounded speed correction from previous muscle state and speed."""

    def __init__(self, muscle_count, *, max_delta_m_s=10.):
        super().__init__()
        if (type(muscle_count) is not int or muscle_count < 1
                or isinstance(max_delta_m_s, bool) or not isinstance(max_delta_m_s, (int, float))
                or not math.isfinite(max_delta_m_s) or not 0 < max_delta_m_s <= 20):
            raise ValueError('Explicit muscle count and bounded speed correction required')
        self.muscle_count = muscle_count
        self.max_delta_m_s = float(max_delta_m_s)
        self.projection = nn.Linear(muscle_count + 1, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, muscles, speed, racing_goals):
        weight = self.projection.weight
        if (not isinstance(muscles, torch.Tensor) or muscles.shape != (1, self.muscle_count)
                or not isinstance(speed, torch.Tensor) or speed.shape != (1, 1)
                or not isinstance(racing_goals, torch.Tensor) or racing_goals.shape != (1, 2)
                or any(t.device != weight.device or t.dtype != weight.dtype
                       or not torch.isfinite(t).all() for t in (muscles, speed, racing_goals))
                or torch.any(speed < 0) or torch.any(racing_goals[:, 0] != 0)
                or torch.any((racing_goals[:, 1] <= 0) | (racing_goals[:, 1] > 3.5))):
            raise ValueError('Finite flight muscles, body speed and racing ceiling required')
        correction = self.max_delta_m_s * torch.tanh(
            self.projection(torch.cat((muscles, speed / 30.), dim=1)))
        target = torch.minimum((speed + correction).clamp_min(0.), racing_goals[:, 1:2] * (100. / 3.6))
        return torch.cat((torch.zeros_like(target), target), dim=1)

class SymmetricFlightStateSpeedCommand(FlightStateSpeedCommand):
    """Experimental bounded symmetric readout of the same six flight muscle states.

    The three original left/right muscle pairs are not a second controller. This
    adapter is an engineering hypothesis and is not qualified for vehicle use.
    """
    def __init__(self, muscle_count, *, max_delta_m_s=10., pair_scale=.001, max_gain=2.):
        if muscle_count != 6:
            raise ValueError('Exact three original left/right muscle pairs required')
        if (isinstance(pair_scale, bool) or not isinstance(pair_scale, (float, int))
                or not math.isfinite(pair_scale) or not .001 <= pair_scale <= 1.
                or isinstance(max_gain, bool) or not isinstance(max_gain, (float, int))
                or not math.isfinite(max_gain) or not 0 < max_gain <= 2.):
            raise ValueError('Explicit bounded pair scale and gain required')
        super().__init__(muscle_count, max_delta_m_s=max_delta_m_s)
        self.pair_scale, self.max_gain = float(pair_scale), float(max_gain)
        self.projection = nn.Linear(4, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, muscles, speed, racing_goals):
        weight = self.projection.weight
        if (not isinstance(muscles, torch.Tensor) or muscles.ndim != 2 or muscles.shape[1] != 6 or len(muscles) == 0
                or not isinstance(speed, torch.Tensor) or speed.shape != (len(muscles), 1)
                or not isinstance(racing_goals, torch.Tensor) or racing_goals.shape != (len(muscles), 2)
                or any(t.device != weight.device or t.dtype != weight.dtype
                       or not torch.isfinite(t).all() for t in (muscles, speed, racing_goals))
                or torch.any(speed < 0) or torch.any(racing_goals[:, 0] != 0)
                or torch.any((racing_goals[:, 1] <= 0) | (racing_goals[:, 1] > 3.5))):
            raise ValueError('Finite flight muscles, body speed and racing ceiling required')
        pair_difference = torch.abs(muscles[:, 1::2] - muscles[:, ::2])
        features = torch.cat((torch.tanh(pair_difference / self.pair_scale), speed / 30.), dim=1)
        bounded_weight = self.max_gain * torch.tanh(self.projection.weight)
        bounded_bias = self.max_gain * torch.tanh(self.projection.bias)
        correction = self.max_delta_m_s * torch.tanh(features @ bounded_weight.T + bounded_bias)
        target = torch.minimum((speed + correction).clamp_min(0.), racing_goals[:, 1:2] * (100. / 3.6))
        return torch.cat((torch.zeros_like(target), target), dim=1)
