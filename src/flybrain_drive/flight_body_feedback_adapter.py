"""Experimental vehicle-body feedback into the existing flight sensory boundary.

Tilt, wheel loads, tyre temperatures and wheel rates are vehicle surrogates, not
measured fly receptor responses. This module adds no actuator or route input.
"""

import math

import torch
from torch import nn

from .flight_embodied_navigation import modulate_current
from .flight_maneuver import MANEUVER_CONTRACT
from .racing_goal import RACING_GOAL_CONTRACT


class VehicleBodyFeedbackAdapter(nn.Module):
    """Bounded, initially neutral masked vehicle-to-sensory-port projection."""

    def __init__(self, sensory_ports):
        super().__init__()
        if type(sensory_ports) is not int or sensory_ports < 1:
            raise ValueError('Explicit number of original sensory input ports required')
        indices = [0, 1, *range(8, 16), *range(17, 21)]
        scales = [.1, .1, *([1.] * 4), *([100.] * 4), *([30.] * 4)]
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.projection = nn.Linear(2*len(indices), sensory_ports, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, body, body_valid):
        if (body.ndim != 2 or body.shape[1] != 21 or len(body) == 0
                or body_valid.shape != body.shape or body_valid.dtype != torch.bool
                or body.device != self.scales.device or body_valid.device != body.device
                or body.dtype != self.scales.dtype or not torch.isfinite(body).all()):
            raise ValueError('Finite body21 and explicit original availability mask required')
        available = body_valid[:, self.indices]
        features = torch.tanh(body[:, self.indices] / self.scales)
        features = torch.where(available, features, torch.zeros_like(features))
        encoded = torch.cat([features, available.to(body.dtype)], dim=1)
        return .5 * torch.tanh(self.projection(encoded))


class CenteredVehicleBodyFeedbackAdapter(VehicleBodyFeedbackAdapter):
    """Availability relative to fully valid input; absent channels remain explicit."""

    def forward(self, body, body_valid):
        if (body.ndim != 2 or body.shape[1] != 21 or len(body) == 0
                or body_valid.shape != body.shape or body_valid.dtype != torch.bool
                or body.device != self.scales.device or body_valid.device != body.device
                or body.dtype != self.scales.dtype or not torch.isfinite(body).all()):
            raise ValueError('Finite body21 and explicit original availability mask required')
        available = body_valid[:, self.indices]
        values = torch.tanh(body[:, self.indices] / self.scales)
        values = torch.where(available, values, torch.zeros_like(values))
        availability_delta = available.to(body.dtype)-1.
        return .5*torch.tanh(self.projection(torch.cat([values, availability_delta], dim=1)))


class CenteredSpeedBodyFeedbackAdapter(CenteredVehicleBodyFeedbackAdapter):
    """Experimental centered body boundary including unsigned speed in m/s.

    The 30 m/s scale is an engineering normalization, not a fly receptor model.
    Zero initialization preserves a neutral added path until explicitly trained.
    """

    def __init__(self, sensory_ports):
        if type(sensory_ports) is not int or sensory_ports < 1:
            raise ValueError('Explicit number of original sensory input ports required')
        nn.Module.__init__(self)
        indices = [0, 1, *range(8, 16), 16, *range(17, 21)]
        scales = [.1, .1, *([1.] * 4), *([100.] * 4), 30., *([30.] * 4)]
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.projection = nn.Linear(2 * len(indices), sensory_ports, bias=False)
        nn.init.zeros_(self.projection.weight)


class ExperimentalBodyFeedbackCircuit(nn.Module):
    """Wrap an original physical circuit without changing its graph or output readout.

    This engineering projection requires separate training and physiological,
    closed-loop, and validity qualification before it can control a vehicle.
    """

    def __init__(self, motor):
        super().__init__()
        if not isinstance(motor, nn.Module) or not hasattr(motor, 'backbone'):
            raise ValueError('Original physical flight motor circuit required')
        self.motor = motor
        self.adapter = VehicleBodyFeedbackAdapter(len(motor.backbone.body_ports))
        self.neurons = motor.neurons

    def initial_state(self, batch):
        return self.motor.initial_state(batch)

    def drive(self, command, body, body_valid, racing_goals, *, command_contract,
              goal_contract):
        if command_contract != MANEUVER_CONTRACT or goal_contract != RACING_GOAL_CONTRACT:
            raise ValueError('Original physical maneuver and racing task contracts required')
        drive = self.motor.drive(command, body, body_valid, racing_goals,
            command_contract=command_contract, goal_contract=goal_contract)
        ports = self.motor.backbone.body_ports
        delta = self.adapter(body, body_valid)
        if drive.shape != (len(body), self.neurons) or delta.shape != (len(body), len(ports)):
            raise ValueError('Original sensory port and drive dimensions required')
        updated = drive.clone()
        updated[:, ports] = modulate_current(drive[:, ports], delta)
        return updated

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
