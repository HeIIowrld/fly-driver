"""Directional time-trace adapter for original HS/VS ports, research only.

T4/T5 direction differences follow the repository's pinned author ordering.
The projection and time scales are engineering hypotheses, not measured synapses.
"""

import torch
from torch import nn

from .flight_temporal_motion_boundary import TemporalMotionBoundary


class OpponentTemporalMotionBoundary(TemporalMotionBoundary):
    """Project separate ON/OFF, opponent-direction and four-view temporal traces."""

    def __init__(self, train_mean, train_scale, port_count, *, fast_seconds=.05,
                 slow_seconds=.2):
        super().__init__(train_mean, train_scale, port_count, fast_seconds=fast_seconds,
                         slow_seconds=slow_seconds)
        # Retain the parent for its validated causal state contract, not its readout.
        self.projection.requires_grad_(False)
        self.opponent_projection = nn.Linear(
            256, port_count, bias=False, device=train_mean.device, dtype=train_mean.dtype
        )
        nn.init.zeros_(self.opponent_projection.weight)

    def step(self, scene, view_valid, state, elapsed):
        _, traces = super().step(scene, view_valid, state, elapsed)
        motion = (traces[0] - traces[1]).reshape(len(scene), 4, 2, 4, 16)
        opponent = torch.stack((motion[:, :, :, 1] - motion[:, :, :, 0],
                                motion[:, :, :, 2] - motion[:, :, :, 3]), dim=3)
        delta = .5 * torch.tanh(self.opponent_projection(opponent.reshape(len(scene), 256)))
        return delta, traces
