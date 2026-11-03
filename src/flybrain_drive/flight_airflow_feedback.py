"""Experimental body-frame relative airflow input for the existing flight sensory ports.

This is an engineering hypothesis, not measured fly sensory physiology. Source values
must be actual relative air velocity, never vehicle ground speed or world wind alone.
"""

import math

import torch
from torch import nn

from .flight_body_feedback_adapter import ExperimentalBodyFeedbackCircuit
from .flight_embodied_navigation import modulate_current


class RelativeAirflowAdapter(nn.Module):
    """Body-frame forward/left/up relative air velocity, m/s, with validity per axis."""

    def __init__(self, sensory_ports):
        super().__init__()
        if type(sensory_ports) is not int or sensory_ports < 1:
            raise ValueError('Explicit sensory port count required')
        self.projection = nn.Linear(6, sensory_ports, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, airflow, valid):
        if (airflow.ndim != 2 or airflow.shape[1] != 3 or len(airflow) < 1
                or valid.shape != airflow.shape or valid.dtype != torch.bool
                or airflow.device != self.projection.weight.device
                or valid.device != airflow.device or airflow.dtype != self.projection.weight.dtype
                or not torch.isfinite(airflow).all()):
            raise ValueError('Finite body-frame airflow and explicit per-axis mask required')
        channels = torch.where(valid, torch.tanh(airflow/30.), torch.zeros_like(airflow))
        return .5*torch.tanh(self.projection(torch.cat((channels, valid.to(airflow.dtype)), 1)))


class ExperimentalAirflowBodyCircuit(ExperimentalBodyFeedbackCircuit):
    """Add optional airflow to the same sensory ports, with original graph and readout.

    The caller validates provenance, sign, units, receipt age, and validity. This class
    does not derive airflow from speed, access simulator truth, or drive actuators.
    """

    def __init__(self, motor):
        super().__init__(motor)
        self.airflow_adapter = RelativeAirflowAdapter(len(motor.backbone.body_ports))

    def drive(self, command, body, body_valid, racing_goals, airflow, airflow_valid, *,
              command_contract, goal_contract):
        drive = super().drive(command, body, body_valid, racing_goals,
                              command_contract=command_contract, goal_contract=goal_contract)
        if len(airflow) != len(body):
            raise ValueError('Airflow and body batches must match')
        delta = self.airflow_adapter(airflow, airflow_valid)
        ports = self.motor.backbone.body_ports
        output = drive.clone()
        output[:, ports] = modulate_current(drive[:, ports], delta)
        return output

    def advance(self, command, body, body_valid, racing_goals, airflow, airflow_valid,
                state, elapsed, *, command_contract, goal_contract):
        if (not math.isfinite(elapsed) or not 0 < elapsed <= .08
                or state.shape != (len(body), self.neurons) or not torch.isfinite(state).all()
                or torch.any((state < 0) | (state > 1))):
            raise ValueError('Bounded state and original positive interval <=80ms required')
        self.motor.backbone.assert_body_boundary_protected()
        drive = self.drive(command, body, body_valid, racing_goals, airflow, airflow_valid,
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
