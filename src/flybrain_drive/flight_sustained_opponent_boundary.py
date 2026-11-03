"""Experimental parallel sustained and transient opponent-motion boundary.

This is an engineering hypothesis, not a measured fly circuit or a driving policy.
The slow trace is normalized relative to TRAIN motion activity; it is not a direct
measurement of optic flow velocity or evidence that a static image is moving.
"""

import torch
from torch import nn

from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary


def opponent_axes(motion):
    """Keep view, ON/OFF, axis and receptor bins; use the pinned T4/T5 ordering."""
    motion = motion.reshape(len(motion), 4, 2, 4, 16)
    return torch.stack((motion[:, :, :, 1] - motion[:, :, :, 0],
                        motion[:, :, :, 2] - motion[:, :, :, 3]), dim=3).reshape(
                            len(motion), 256)


class SustainedOpponentBoundary(OpponentTemporalMotionBoundary):
    """Preserve the existing transient path and add a zero-initialized slow path."""

    def __init__(self, train_mean, train_scale, port_count, **kwargs):
        super().__init__(train_mean, train_scale, port_count, **kwargs)
        self.sustained_projection = nn.Linear(256, port_count, bias=False,
                                              device=train_mean.device, dtype=train_mean.dtype)
        nn.init.zeros_(self.sustained_projection.weight)

    def step(self, scene, view_valid, state, elapsed):
        _, traces = super().step(scene, view_valid, state, elapsed)
        transient = opponent_axes(traces[0] - traces[1])
        sustained = opponent_axes(traces[1])
        logits = self.opponent_projection(transient) + self.sustained_projection(sustained)
        return .5 * torch.tanh(logits), traces
