import pytest
import torch
from torch import nn

from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.flight_stop_task_adapter import ExperimentalStopTaskCircuit, StopTaskDNAdapter
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT


class FakeMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.register_buffer('goal_ports', torch.tensor([1, 3]))
        self.neurons = 5

    def drive(self, command, body, body_valid, racing_goals, **kwargs):
        return body.new_tensor([[.1, .2, .3, .4, .5]]).expand(len(body), -1).clone()


def test_stop_modulation_starts_at_exact_identity_and_changes_only_goal_ports():
    model = ExperimentalStopTaskCircuit(FakeMotor())
    body, mask = torch.zeros(2, 21), torch.ones(2, 21, dtype=torch.bool)
    command, goals = torch.tensor([[0., 0.], [0., 20.]]), torch.ones(2, 2)
    kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    original = model.motor.drive(command, body, mask, goals, **kwargs)
    assert torch.equal(model.drive(command, body, mask, goals, **kwargs), original)
    with torch.no_grad():
        model.adapter.stop_delta.fill_(.5)
    changed = model.drive(command, body, mask, goals, **kwargs)
    assert torch.equal(changed[:, [0, 2, 4]], original[:, [0, 2, 4]])
    assert torch.all(changed[:, [1, 3]] >= original[:, [1, 3]])
    assert (changed[0, 1]-original[0, 1]) > 1000*(changed[1, 1]-original[1, 1])


def test_stop_signal_is_differentiable_and_invalid_goal_rejected():
    adapter = StopTaskDNAdapter()
    adapter(torch.tensor([0.])).sum().backward()
    assert adapter.stop_delta.grad.item() != 0
    with pytest.raises(ValueError, match='nonnegative'):
        adapter(torch.tensor([-1.]))
    with pytest.raises(ValueError, match='nonnegative'):
        adapter(torch.tensor([float('nan')]))
