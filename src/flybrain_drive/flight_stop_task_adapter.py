"""Experimental stop-task modulation of existing descending flight inputs.

This is a learned engineering task-to-current hypothesis, not measured fly
anatomy or a direct brake actuator. It requires independent qualification.
"""

import math

import torch
from torch import nn

from .flight_embodied_navigation import modulate_current
from .flight_maneuver import MANEUVER_CONTRACT
from .racing_goal import RACING_GOAL_CONTRACT


class StopTaskDNAdapter(nn.Module):
    """Bounded zero-initialized goal-current modulation from declared target speed."""

    def __init__(self):
        super().__init__()
        self.stop_delta = nn.Parameter(torch.zeros(()))

    def forward(self, desired_speed_m_s):
        if (desired_speed_m_s.ndim != 1 or len(desired_speed_m_s) == 0
                or desired_speed_m_s.device != self.stop_delta.device
                or desired_speed_m_s.dtype != self.stop_delta.dtype
                or not torch.isfinite(desired_speed_m_s).all()
                or torch.any(desired_speed_m_s < 0)):
            raise ValueError('Finite declared nonnegative target speed required')
        gate = torch.exp(-desired_speed_m_s / 2.)
        return 2*torch.tanh(self.stop_delta)*gate[:, None]


class ExperimentalStopTaskCircuit(nn.Module):
    """Wrap physical motor graph without adding a direct vehicle brake output."""

    def __init__(self, motor):
        super().__init__()
        if not isinstance(motor, nn.Module) or not hasattr(motor, 'backbone'):
            raise ValueError('Original physical motor circuit required')
        self.motor = motor
        self.adapter = StopTaskDNAdapter()
        self.neurons = motor.neurons

    def drive(self, command, body, body_valid, racing_goals, *, command_contract,
              goal_contract):
        if command_contract != MANEUVER_CONTRACT or goal_contract != RACING_GOAL_CONTRACT:
            raise ValueError('Original physical maneuver and racing task contracts required')
        original = self.motor.drive(command, body, body_valid, racing_goals,
            command_contract=command_contract, goal_contract=goal_contract)
        ports = self.motor.backbone.goal_ports
        if original.shape != (len(body), self.neurons) or command.shape != (len(body), 2):
            raise ValueError('Original physical command and neural drive dimensions required')
        delta = self.adapter(command[:, 1])
        drive = original.clone()
        drive[:, ports] = modulate_current(original[:, ports], delta)
        return drive

    def advance(self, command, body, body_valid, racing_goals, state, elapsed, *,
                command_contract, goal_contract):
        if (not math.isfinite(elapsed) or not 0 < elapsed <= .08
                or state.shape != (len(body), self.neurons) or not torch.isfinite(state).all()
                or torch.any((state < 0) | (state > 1))):
            raise ValueError('Bounded state and original positive interval <=80ms required')
        self.motor.backbone.assert_body_boundary_protected()
        drive = self.drive(command, body, body_valid, racing_goals,
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
