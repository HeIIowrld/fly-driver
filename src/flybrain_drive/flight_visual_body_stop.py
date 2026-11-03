"""Experimental joint sensory circuit with a separate stop-goal boundary.

The stop adapter is an engineering hypothesis, not validated fly physiology or
an actuator policy. Its weights require graph/port provenance before any transfer.
"""

import torch

from .flight_embodied_navigation import modulate_current
from .flight_stop_narrow_adapter import NarrowStopVectorAdapter
from .flight_stop_vector_adapter import StopVectorAdapter
from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow


class ExperimentalVisualBodyStop(ExperimentalVisualBodyAirflow):
    """Add an initially neutral speed-goal modulation on the original DNg ports."""

    def __init__(self, circuit, *, centered_availability=False):
        super().__init__(circuit, centered_availability=centered_availability)
        self.stop = StopVectorAdapter(len(circuit.backbone.goal_ports))

    def drive(self, visual, visual_valid, body, body_valid, goals, scene_features,
              scene_valid, airflow, airflow_valid):
        original = super().drive(visual, visual_valid, body, body_valid, goals, scene_features,
                                 scene_valid, airflow, airflow_valid)
        # v7's speed fraction is a declared 0..30 km/h task, not measured vehicle speed.
        speed_m_s = goals[:, 1] * (30. / 3.6)
        stop_delta = self.stop(speed_m_s)
        ports = self.circuit.backbone.goal_ports
        drive = original.clone()
        drive[:, ports] = modulate_current(original[:, ports], stop_delta)
        return drive


class ExperimentalVisualBodyNarrowStop(ExperimentalVisualBodyStop):
    """Narrow 0.1 m/s goal gate; changes low-speed response, not an actor policy."""

    def __init__(self, circuit, *, centered_availability=False):
        super().__init__(circuit, centered_availability=centered_availability)
        self.stop = NarrowStopVectorAdapter(len(circuit.backbone.goal_ports))


class ExperimentalVisualBodySpeedErrorStop(ExperimentalVisualBodyStop):
    """Provisional speed-feedback gate; constants are untrained engineering hypotheses."""

    def drive(self, visual, visual_valid, body, body_valid, goals, scene_features,
              scene_valid, airflow, airflow_valid):
        if (body.ndim != 2 or body.shape[1] != 21 or body_valid.shape != body.shape
                or body_valid.dtype != torch.bool or not body_valid[:, 16].all()
                or not torch.isfinite(body[:, 16]).all() or torch.any(body[:, 16] < 0)):
            raise ValueError('Valid unsigned vehicle speed in m/s is required')
        original = ExperimentalVisualBodyAirflow.drive(
            self, visual, visual_valid, body, body_valid, goals, scene_features,
            scene_valid, airflow, airflow_valid)
        target = goals[:, 1] * (30. / 3.6)
        if not torch.isfinite(target).all() or torch.any(target < 0):
            raise ValueError('Finite nonnegative declared speed goal required')
        error = (body[:, 16] - target).clamp_min(0)
        stopped_goal = torch.exp(-target / .1)
        overspeed = torch.exp(-target / 2.) * torch.tanh(error / 2.)
        gate = torch.maximum(stopped_goal, overspeed)[:, None]
        delta = 2 * torch.tanh(self.stop.stop_delta)[None, :] * gate
        result = original.clone()
        ports = self.circuit.backbone.goal_ports
        result[:, ports] = modulate_current(original[:, ports], delta)
        return result
