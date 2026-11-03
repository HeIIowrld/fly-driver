"""Experimental normalized wheel-load feedback into existing flight body ports.

Native wheel ordering is deliberately not interpreted as left/right or front/rear.
This is an engineering surrogate, not a measured fly receptor or real actor.
"""

import torch
from torch import nn

from .synthetic_acceleration_feedback import SyntheticAccelerationBodyFeedbackAdapter


class SyntheticLoadBodyFeedbackAdapter(SyntheticAccelerationBodyFeedbackAdapter):
    """Add an initially neutral four-wheel imbalance path to the same body ports.

    Indices 8:12 are normalized load fractions in native order. All four must
    be available together; a partial group contributes zero load values to both
    projections. Original availability indicators are retained, as are all
    other body features and the synthetic acceleration path.
    """

    def __init__(self, sensory_ports):
        super().__init__(sensory_ports)
        self.load_projection = nn.Linear(4, sensory_ports, bias=False)
        nn.init.zeros_(self.load_projection.weight)

    def forward(self, body, body_valid):
        if (not isinstance(body, torch.Tensor) or body.ndim != 2 or body.shape[1] != 21
                or len(body) == 0 or not isinstance(body_valid, torch.Tensor)
                or body_valid.shape != body.shape or body_valid.dtype != torch.bool
                or body.device != self.scales.device or body_valid.device != body.device
                or body.dtype != self.scales.dtype or not torch.isfinite(body).all()):
            raise ValueError('Finite body21 and explicit original availability mask required')
        available = body_valid[:, self.indices]
        load = body[:, 8:12]
        complete = body_valid[:, 8:12].all(dim=1, keepdim=True)
        value_mask = available.clone()
        value_mask[:, 2:6] &= complete
        values = torch.tanh(body[:, self.indices] / self.scales)
        values = torch.where(value_mask, values, torch.zeros_like(values))
        encoded = torch.cat([values, available.to(body.dtype)-1.], dim=1)
        accel = torch.tanh(body[:, 5:8] / self.acceleration_scales)
        accel = torch.where(body_valid[:, 5:8], accel, torch.zeros_like(accel))
        malformed = complete & (((load < 0) | (load > 1)).any(dim=1, keepdim=True)
                                | ((load.sum(dim=1, keepdim=True) - 1).abs() > 1e-4))
        if malformed.any():
            raise ValueError('Available native load fractions must be nonnegative and sum to one')
        imbalance = torch.where(complete, load - .25, torch.zeros_like(load))
        return .5 * torch.tanh(self.projection(encoded) + self.acceleration_projection(accel)
                                + self.load_projection(imbalance))
