import pytest
import torch
from torch import nn

from flybrain_drive.flight_body_feedback_adapter import (
    ExperimentalBodyFeedbackCircuit,
    VehicleBodyFeedbackAdapter,
)
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('body_ports', torch.tensor([1, 3]))
        self.spec = {'dynamics': {'dt': .01}}

    def assert_body_boundary_protected(self):
        pass

    def _advance_elapsed_tensor(self, drive, state, elapsed):
        return state + elapsed*drive

    def _read_state_tensor(self, state):
        return state[:, :2], state[:, 2:]


class FakeMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.neurons = 5

    def initial_state(self, batch):
        return torch.zeros(batch, self.neurons)

    def drive(self, command, body, body_valid, racing_goals, **kwargs):
        return body.new_tensor([[.1, .2, .3, .4, .5]]).expand(len(body), -1).clone()


def test_initial_identity_and_masked_unavailable_values():
    motor = FakeMotor()
    circuit = ExperimentalBodyFeedbackCircuit(motor)
    body = torch.zeros(2, 21)
    body[:, 0] = .1
    body[:, 8] = .3
    valid = torch.ones_like(body, dtype=torch.bool)
    command, goals = torch.zeros(2, 2), torch.ones(2, 2)
    kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    baseline = motor.drive(command, body, valid, goals, **kwargs)
    actual = circuit.drive(command, body, valid, goals, **kwargs)
    assert torch.equal(baseline, actual)
    with torch.no_grad():
        circuit.adapter.projection.weight[0, 0] = 1
        circuit.adapter.projection.weight[1, 2] = 1
    changed = circuit.drive(command, body, valid, goals, **kwargs)
    assert torch.equal(changed[:, [0, 2, 4]], baseline[:, [0, 2, 4]])
    assert torch.all(changed[:, [1, 3]] > baseline[:, [1, 3]])
    valid[:, 0] = False
    body[0, 0], body[1, 0] = 5., -5.
    masked = circuit.drive(command, body, valid, goals, **kwargs)
    assert torch.equal(masked[0], masked[1])


def test_original_time_step_and_sensory_gradient_are_retained():
    circuit = ExperimentalBodyFeedbackCircuit(FakeMotor())
    body = torch.zeros(1, 21)
    body[:, 1] = .1
    mask = torch.ones_like(body, dtype=torch.bool)
    kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    intent, _, state = circuit.advance(torch.zeros(1, 2), body, mask, torch.ones(1, 2),
                                       circuit.initial_state(1), .025, **kwargs)
    torch.testing.assert_close(state, torch.tensor([[.0025, .005, .0075, .01, .0125]]))
    intent.sum().backward()
    assert circuit.adapter.projection.weight.grad.abs().sum() > 0


def test_invalid_mask_and_time_are_rejected():
    adapter = VehicleBodyFeedbackAdapter(2)
    body = torch.zeros(1, 21)
    with pytest.raises(ValueError, match='availability'):
        adapter(body, torch.ones_like(body))
    circuit = ExperimentalBodyFeedbackCircuit(FakeMotor())
    with pytest.raises(ValueError, match='interval'):
        circuit.advance(torch.zeros(1, 2), body, torch.ones_like(body, dtype=torch.bool),
                        torch.ones(1, 2), circuit.initial_state(1), .081,
                        command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)


def test_centered_availability_does_not_create_constant_drive_but_marks_missing():
    from flybrain_drive.flight_body_feedback_adapter import CenteredVehicleBodyFeedbackAdapter

    adapter = CenteredVehicleBodyFeedbackAdapter(2)
    with torch.no_grad():
        adapter.projection.weight[:, 14:].fill_(1.)
    body = torch.zeros(2, 21)
    valid = torch.ones_like(body, dtype=torch.bool)
    assert torch.equal(adapter(body, valid), torch.zeros(2, 2))
    valid[0, 0] = False
    assert (adapter(body, valid)[0] < 0).all()
    assert (adapter(body, valid)[1] == 0).all()


def test_centered_speed_boundary_tracks_speed_and_preserves_missingness():
    from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter

    adapter = CenteredSpeedBodyFeedbackAdapter(1)
    index = adapter.indices.tolist().index(16)
    with torch.no_grad():
        adapter.projection.weight[0, index] = 1.
        adapter.projection.weight[0, len(adapter.indices) + index] = 1.
    body = torch.zeros(2, 21)
    body[1, 16] = 15.
    valid = torch.ones_like(body, dtype=torch.bool)
    output = adapter(body, valid)
    assert output[0, 0] == 0 and output[1, 0] > 0
    valid[:, 16] = False
    missing = adapter(body, valid)
    assert missing[0, 0] == missing[1, 0] and missing[0, 0] < 0
