"""Physical maneuver boundary into measured flight topology, offline diagnostics only.

These currents are engineering encodings, not measured fly tuning. The legacy
30km/h task encoder is not called. There is no live actuator interface here.
"""

import math

import torch
from torch import nn

from .flight_commands import COMMAND_GOAL_CONTRACT, SeparatedFlightCircuit
from .graph import sha256
from .racing_goal import RACING_GOAL_CONTRACT

MANEUVER_CONTRACT = {
    'schema': 'flybrain.flight.physical_maneuver.v1',
    'fields': ['desired_yaw_left_rad_s', 'desired_speed_m_s'],
    'maximum_absolute_yaw_rad_s': 2.,
    'yaw_scale_rad_s': 1.,
    'speed_error_scale_m_s': 10.,
    'source': 'Vision/body actor estimate; oracle inputs restricted to labelled diagnostics',
    'external_goal': RACING_GOAL_CONTRACT,
    'yaw_feedback': 'Raw yaw enters protected haltere boundary once; no external yaw subtraction',
    'speed_feedback': 'desired_speed minus unsigned measured speed, once at DNg02 boundary',
    'biological_claim': 'Engineering physical-unit command encoding; no measured DN tuning',
    'checkpoint_compatibility': 'New encoding; never reinterpret legacy speed_fraction as m/s',
    'tensor_precision': 'Actor tensors match the circuit floating dtype; preserve original records',
}


class ManeuverCurrents(nn.Module):
    """Separate turn population and speed-error current; no motor readout."""

    def __init__(self, orientations, power_ports, *, turn_gain=.2, power_gain=2., tonic=.05):
        super().__init__()
        if (len(orientations) != 4 or any(s not in (-1, 1) for s in orientations)
                or type(power_ports) is not int or power_ports < 1
                or not all(math.isfinite(x) for x in (turn_gain, power_gain, tonic))
                or not 0 < turn_gain <= 1 or not 0 < power_gain <= 10 or not 0 < tonic < 1):
            raise ValueError('Explicit oriented DN population and finite current gains required')
        self.register_buffer('orientations', torch.tensor(orientations, dtype=torch.float32))
        self.register_buffer('tonic_logit', torch.logit(torch.tensor(float(tonic))))
        self.turn_gain, self.power_gain, self.power_ports = turn_gain, power_gain, power_ports

    def forward(self, command, body, body_valid, racing_goals, *, command_contract,
                goal_contract, lesion=None):
        if command_contract != MANEUVER_CONTRACT or goal_contract != RACING_GOAL_CONTRACT:
            raise ValueError('Explicit physical-maneuver and racing-ceiling contracts required')
        n = len(body)
        if (n < 1 or command.shape != (n, 2) or body.shape != (n, 21)
                or body_valid.shape != body.shape or body_valid.dtype != torch.bool
                or racing_goals.shape != (n, 2)
                or not all(x.dtype == self.tonic_logit.dtype for x in (command, body, racing_goals))
                or not all(x.device == self.tonic_logit.device
                           for x in (command, body, body_valid, racing_goals))
                or not all(torch.isfinite(x).all() for x in (command, body, racing_goals))
                or not body_valid[:, [4, 16]].all() or torch.any(body[:, 16] < 0)
                or torch.any(command[:, 0].abs() > 2) or torch.any(command[:, 1] < 0)
                or torch.any(racing_goals[:, 0] != 0)
                or torch.any((racing_goals[:, 1] <= 0) | (racing_goals[:, 1] > 3.5))
                or torch.any(command[:, 1] > racing_goals[:, 1]*100/3.6+1e-5)):
            raise ValueError('Physical yaw/speed, original body21 and external ceiling required')
        if lesion not in (None, 'desired_yaw', 'speed_error', 'yaw_feedback'):
            raise ValueError('Explicit maneuver-path intervention required')
        yaw = torch.tanh(command[:, :1]/MANEUVER_CONTRACT['yaw_scale_rad_s'])
        if lesion == 'desired_yaw':
            yaw = torch.zeros_like(yaw)
        steering = self.turn_gain*(yaw*self.orientations).relu()
        error = (command[:, 1:2]-body[:, 16:17])/MANEUVER_CONTRACT['speed_error_scale_m_s']
        if lesion == 'speed_error':
            error = torch.zeros_like(error)
        power = (self.tonic_logit+self.power_gain*torch.tanh(error)).sigmoid()
        return steering, power.expand(-1, self.power_ports)


class PhysicalManeuverCircuit(nn.Module):
    """New physical inputs, original protected body encoder and fixed chemical graph.

    Optional fast-haltere dynamics and old scene/goal encoders are not silently
    inherited. Their separate component certificates do not cover this interface.
    """

    def __init__(self, source_spec, payload, *, sparse_backend='cached_csr_gpu'):
        super().__init__()
        if (payload.get('schema') != 'flybrain.flight.separated_commands.v5'
                or payload.get('spec_sha256') != sha256(source_spec)
                or payload.get('goal_contract') != COMMAND_GOAL_CONTRACT):
            raise ValueError('Explicit original v5 topology/body/readout source required')
        self.backbone = SeparatedFlightCircuit(source_spec,
            glutamate_sign=payload['glutamate_sign'], sparse_backend=sparse_backend)
        self.backbone.load_state_dict(payload['model'], strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.assert_body_boundary_protected()
        self.command = ManeuverCurrents(self.backbone.spec['steering_orientations'],
                                        len(self.backbone.goal_ports))
        self.contract = MANEUVER_CONTRACT
        self.neurons = self.backbone.neurons

    def initial_state(self, batch):
        return self.backbone.initial_state(batch)

    def drive(self, command, body, body_valid, racing_goals, *, command_contract,
              goal_contract, lesion=None):
        steering, power = self.command(command, body, body_valid, racing_goals,
            command_contract=command_contract, goal_contract=goal_contract, lesion=lesion)
        angular = body[:, 2:5]
        if lesion == 'yaw_feedback':
            angular = angular.clone()
            angular[:, 2] = 0  # Preserve original availability; values-only intervention.
        encoded = torch.cat([angular.tanh()*body_valid[:, 2:5], body_valid[:, 2:5].float()], 1)
        drive = body.new_zeros((len(body), self.neurons))
        drive[:, self.backbone.body_ports] = self.backbone.body(encoded).sigmoid()
        drive[:, self.backbone.steering_ports] = steering
        drive[:, self.backbone.goal_ports] = power
        return drive

    def advance(self, command, body, body_valid, racing_goals, state, elapsed, *, command_contract,
                goal_contract, lesion=None):
        if (not math.isfinite(elapsed) or not 0 < elapsed <= .08
                or state.shape != (len(body), self.neurons) or not torch.isfinite(state).all()
                or torch.any((state < 0) | (state > 1))):
            raise ValueError('Bounded state and original positive interval <=80ms required')
        self.backbone.assert_body_boundary_protected()
        drive = self.drive(command, body, body_valid, racing_goals,
            command_contract=command_contract, goal_contract=goal_contract, lesion=lesion)
        nominal = self.backbone.spec['dynamics']['dt']
        count = int(math.floor(elapsed/nominal))
        for _ in range(count):
            state = self.backbone._advance_elapsed_tensor(drive, state, nominal)
        tail = elapsed-count*nominal
        if tail > 1e-12:
            state = self.backbone._advance_elapsed_tensor(drive, state, tail)
        intent, muscles = self.backbone._read_state_tensor(state)
        return intent, muscles, state
