"""Offline stop-task hysteresis at the flight sensory boundary, not a fly measurement."""

import math

import torch

from .flight_embodied_navigation import modulate_current
from .flight_stop_vector_adapter import ExperimentalStopVectorCircuit


class StopGoalHysteresis:
    """Remember stop request until a declared release speed; reset only at episode start.

    Thresholds are engineering hypotheses: a 0.05 m/s request while stopped remains
    stopped until target reaches 0.1 m/s. This is not a physiological claim.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stopping = False

    def update(self, desired_speed_m_s):
        if (isinstance(desired_speed_m_s, bool) or not isinstance(desired_speed_m_s, (int, float))
                or not math.isfinite(desired_speed_m_s) or desired_speed_m_s < 0):
            raise ValueError("Finite nonnegative target speed required")
        if desired_speed_m_s <= .02:
            self.stopping = True
        elif desired_speed_m_s >= .1:
            self.stopping = False
        return float(self.stopping)


class ExperimentalHystereticStopCircuit(ExperimentalStopVectorCircuit):
    """Apply explicit gate to existing descending ports; no actuator or recurrent reset."""

    def drive(self, command, body, body_valid, racing_goals, stop_gate, *,
              command_contract, goal_contract):
        original = self.motor.drive(command, body, body_valid, racing_goals,
                                    command_contract=command_contract, goal_contract=goal_contract)
        if (stop_gate.shape != (len(body),) or stop_gate.dtype != body.dtype
                or stop_gate.device != body.device or not torch.isfinite(stop_gate).all()
                or torch.any((stop_gate != 0) & (stop_gate != 1))):
            raise ValueError("Explicit binary stop-state per sample required")
        delta = 2 * torch.tanh(self.adapter.stop_delta)[None, :] * stop_gate[:, None]
        result = original.clone()
        ports = self.motor.backbone.goal_ports
        result[:, ports] = modulate_current(original[:, ports], delta)
        return result

    def advance(self, command, body, body_valid, racing_goals, stop_gate, state, elapsed, *,
                command_contract, goal_contract):
        if (not math.isfinite(elapsed) or not 0 < elapsed <= .08
                or state.shape != (len(body), self.neurons) or not torch.isfinite(state).all()
                or torch.any((state < 0) | (state > 1))):
            raise ValueError("Finite original state and bounded elapsed interval required")
        self.motor.backbone.assert_body_boundary_protected()
        drive = self.drive(command, body, body_valid, racing_goals, stop_gate,
                           command_contract=command_contract, goal_contract=goal_contract)
        nominal = self.motor.backbone.spec['dynamics']['dt']
        count = int(math.floor(elapsed/nominal))
        for _ in range(count):
            state = self.motor.backbone._advance_elapsed_tensor(drive, state, nominal)
        tail = elapsed-count*nominal
        if tail > 1e-12:
            state = self.motor.backbone._advance_elapsed_tensor(drive, state, tail)
        intent, muscles = self.motor.backbone._read_state_tensor(state)
        return intent, muscles, state
