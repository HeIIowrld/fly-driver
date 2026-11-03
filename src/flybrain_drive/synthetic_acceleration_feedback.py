"""Experimental acceleration feedback to the existing flight sensory ports.

Engineering hypothesis only: no native axis/gravity registration, physiological
claim, actuator qualification, or permission to train on real recordings.
"""
import torch
from torch import nn

from .flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter


class SyntheticAccelerationBodyFeedbackAdapter(CenteredSpeedBodyFeedbackAdapter):
    """Preserve the old projection and add masked synthetic acceleration columns.

    Inputs are forward/left/up in m/s², already in the 21-value body contract.
    The new projection is initially zero, so all old checkpoints retain exactly
    the same numerical output. A valid zero and a missing value are both neutral
    for the added path; the original body-valid mask is not modified.
    """

    def __init__(self, sensory_ports):
        super().__init__(sensory_ports)
        self.acceleration_projection = nn.Linear(3, sensory_ports, bias=False)
        nn.init.zeros_(self.acceleration_projection.weight)
        self.register_buffer('acceleration_scales', torch.full((3,), 9.80665))

    def forward(self, body, body_valid):
        if (not isinstance(body, torch.Tensor) or body.ndim != 2 or body.shape[1] != 21
                or len(body) == 0 or not isinstance(body_valid, torch.Tensor)
                or body_valid.shape != body.shape or body_valid.dtype != torch.bool
                or body.device != self.scales.device or body_valid.device != body.device
                or body.dtype != self.scales.dtype or not torch.isfinite(body).all()):
            raise ValueError('Finite body21 and explicit original availability mask required')
        available = body_valid[:, self.indices]
        values = torch.tanh(body[:, self.indices] / self.scales)
        values = torch.where(available, values, torch.zeros_like(values))
        encoded = torch.cat([values, available.to(body.dtype)-1.], dim=1)
        accel = torch.tanh(body[:, 5:8] / self.acceleration_scales)
        accel = torch.where(body_valid[:, 5:8], accel, torch.zeros_like(accel))
        return .5 * torch.tanh(self.projection(encoded) + self.acceleration_projection(accel))
