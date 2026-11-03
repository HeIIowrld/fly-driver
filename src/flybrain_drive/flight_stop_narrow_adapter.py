"""Offline experimental zero-speed-selective stop boundary; no vehicle control."""

import torch

from .flight_stop_vector_adapter import ExperimentalStopVectorCircuit, StopVectorAdapter


class NarrowStopVectorAdapter(StopVectorAdapter):
    """A 0.1 m/s engineering gate; not a measured fly receptor or decision rule."""

    def forward(self, desired_speed_m_s):
        if (desired_speed_m_s.ndim != 1 or len(desired_speed_m_s) == 0
                or desired_speed_m_s.device != self.stop_delta.device
                or desired_speed_m_s.dtype != self.stop_delta.dtype
                or not torch.isfinite(desired_speed_m_s).all()
                or torch.any(desired_speed_m_s < 0)):
            raise ValueError("Finite declared nonnegative target speed required")
        gate = torch.exp(-desired_speed_m_s / .1)[:, None]
        return 2 * torch.tanh(self.stop_delta)[None, :] * gate


class ExperimentalNarrowStopCircuit(ExperimentalStopVectorCircuit):
    """Keep the original motor graph and ports; change only experimental task gate."""

    def __init__(self, motor):
        super().__init__(motor)
        self.adapter = NarrowStopVectorAdapter(len(motor.backbone.goal_ports))
