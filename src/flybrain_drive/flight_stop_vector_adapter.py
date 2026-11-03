"""Experimental per-descending-port stop task encoding; not measured fly tuning."""

import torch
from torch import nn

from .flight_stop_task_adapter import ExperimentalStopTaskCircuit


class StopVectorAdapter(nn.Module):
    """Bounded, neutral-initialized engineering stop currents per existing port."""

    def __init__(self, ports):
        super().__init__()
        if type(ports) is not int or ports < 1:
            raise ValueError("Explicit existing descending-port count required")
        self.stop_delta = nn.Parameter(torch.zeros(ports))

    def forward(self, desired_speed_m_s):
        if (desired_speed_m_s.ndim != 1 or len(desired_speed_m_s) == 0
                or desired_speed_m_s.device != self.stop_delta.device
                or desired_speed_m_s.dtype != self.stop_delta.dtype
                or not torch.isfinite(desired_speed_m_s).all()
                or torch.any(desired_speed_m_s < 0)):
            raise ValueError("Finite declared nonnegative target speed required")
        gate = torch.exp(-desired_speed_m_s / 2.)[:, None]
        return 2 * torch.tanh(self.stop_delta)[None, :] * gate


class ExperimentalStopVectorCircuit(ExperimentalStopTaskCircuit):
    """Preserve original graph, readout and stop-task contract with per-port signals."""

    def __init__(self, motor):
        super().__init__(motor)
        self.adapter = StopVectorAdapter(len(motor.backbone.goal_ports))
