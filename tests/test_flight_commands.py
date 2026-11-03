import torch
from torch import nn

from flybrain_drive.flight_commands import FrozenLinear, PowerGoal, SteeringCommand


def test_speed_boundary_has_no_lateral_task_input():
    power = PowerGoal(1, 3)
    with torch.no_grad():
        power.weight.fill_(.4)
        power.bias.zero_()
    goals = torch.tensor([[-.8, .5], [.8, .5], [.8, 1.]], requires_grad=True)
    result = power(goals)
    torch.testing.assert_close(result[0], result[1])
    assert (result[2] > result[1]).all()
    result.sum().backward()
    assert torch.count_nonzero(goals.grad[:, 0]) == 0


def test_turn_population_currents_ignore_speed_and_preserve_zero_command():
    steering = SteeringCommand([-1, 1, -1, 1])
    goals = torch.tensor([[.5, .5], [.5, 1.], [-.5, .5], [0., .5]])
    currents = steering(goals)
    torch.testing.assert_close(currents[0], currents[1])
    torch.testing.assert_close(currents[0], torch.tensor([0, .1, 0, .1]))
    torch.testing.assert_close(currents[2], torch.tensor([.1, 0, .1, 0]))
    assert torch.count_nonzero(currents[3]) == 0
    assert (currents >= 0).all() and (currents <= 1).all()


def test_frozen_body_survives_adam_momentum_and_weight_decay_with_real_gradients():
    torch.manual_seed(17)
    frozen = FrozenLinear(torch.randn(3, 6), torch.randn(3))
    network = nn.Sequential(frozen, nn.Linear(3, 2))
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    head_before = network[1].weight.detach().clone()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.01, weight_decay=.3)
    inputs = torch.randn(8, 6, requires_grad=True)
    for _ in range(4):
        optimizer.zero_grad()
        loss = network(inputs).square().mean()
        loss.backward()
        optimizer.step()
    assert inputs.grad.abs().sum() > 0
    assert not torch.equal(head_before, network[1].weight)
    assert list(frozen.parameters()) == []
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
