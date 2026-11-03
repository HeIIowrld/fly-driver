import copy

import numpy as np
import pytest
import torch

from scripts.train_flight_visual_guarded import guarded_step, objective_pass


def test_guarded_adam_backtracks_before_committing_an_oversized_step():
    parameter = torch.nn.Parameter(torch.tensor(.5))
    optimizer = torch.optim.Adam([parameter], lr=10.)
    parameter.square().backward()
    report = guarded_step(optimizer, [parameter], lambda: float(parameter.square().detach()),
                          .25, trials=8, contraction=.5, armijo=.0001)
    assert report["accepted"]
    assert not report["trials"][0]["accepted"]
    assert report["trials"][-1]["fraction"] < 1
    assert parameter.square() < .25
    assert optimizer.state[parameter]["step"] == 1


def test_rejected_step_restores_existing_adam_moments_and_parameter_exactly():
    parameter = torch.nn.Parameter(torch.tensor(.5))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    parameter.square().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameter.square().backward()
    before = parameter.detach().clone()
    moments = copy.deepcopy(optimizer.state[parameter])
    report = guarded_step(optimizer, [parameter], lambda: 10., float(parameter.square().detach()),
                          trials=3, contraction=.5, armijo=.0001)
    assert not report["accepted"]
    torch.testing.assert_close(parameter, before, atol=0, rtol=0)
    for key, value in moments.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value, atol=0, rtol=0)


def test_objective_failure_restores_weights_and_does_not_create_adam_state():
    parameter = torch.nn.Parameter(torch.tensor(.5))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    parameter.square().backward()

    def failed_objective():
        raise RuntimeError("Missing source observation")

    with pytest.raises(RuntimeError, match="source observation"):
        guarded_step(optimizer, [parameter], failed_objective,
                     .25, trials=3, contraction=.5, armijo=.0001)
    torch.testing.assert_close(parameter, torch.tensor(.5), atol=0, rtol=0)
    assert not optimizer.state


def test_full_history_objective_weights_labels_and_preserves_unlabelled_prefix(monkeypatch):
    class TinyCausalModel:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(.2))

        def initial_state(self, batch):
            return torch.zeros(batch, 1)

        def _advance_elapsed_tensor(self, held, state, dt):
            return .9 * state + held * dt

        def _read_state_preactivation_tensor(self, state):
            return torch.cat([state, torch.zeros_like(state)], 1), None

    model = TinyCausalModel()
    x = torch.tensor([[2.], [3.], [5.], [7.], [11.], [13.], [17.]])
    target = torch.tensor([[0., 0.], [0., 0.], [.1, 0.], [.2, 0.],
                           [-.3, 0.], [0., 0.], [.4, 0.]])
    valid = torch.tensor([[False, False], [False, False], [True, True], [True, True],
                          [True, True], [False, False], [True, True]])
    schedule = [[], [.01], [.01, .005], [.02], [.01], [.007], [.02, .003]]
    data = dict(x=x, target=target, target_valid=valid)
    episodes = [({}, data, schedule, dict(target_valid=valid.numpy(), target=target.numpy()))]
    monkeypatch.setattr('scripts.train_flight_visual_guarded.drives_for',
                        lambda m, d, start, stop: m.weight * d['x'][start:stop])
    # Independent scalar recurrence includes the two unlabelled prefix observations.
    state, states = model.weight * 0, [model.weight * 0]
    for index in range(1, len(x)):
        for dt in schedule[index]:
            state = .9 * state + model.weight * x[index - 1, 0] * dt
        states.append(state)
    z = torch.stack(states)[valid[:, 0]]
    y = (target[valid[:, 0], 0] + 1) / 2
    reference = (torch.nn.functional.softplus(2 * z) - 2 * z * y).mean() / .01
    reference.backward()
    expected_gradient = model.weight.grad.clone()
    for chunk in [2, 3, 7]:
        with torch.no_grad():
            measured, _ = objective_pass(model, episodes, chunk)
        assert measured == pytest.approx(float(reference.detach()), abs=1e-5)
    model.weight.grad = None
    measured, _ = objective_pass(model, episodes, 7, backward=True)
    torch.testing.assert_close(model.weight.grad, expected_gradient)
    assert np.isfinite(measured)
    # A different cost-sensitive objective still counts every observed label once,
    # and uses each stratum's original total rather than a chunk-local class count.
    target[3, 0] = 0.
    factors = torch.stack(states).detach()[valid[:, 0]].double() / model.weight.detach().double()
    z = model.weight.detach().double() * factors
    t = target[valid[:, 0], 0].double()
    losses = (torch.nn.functional.softplus(2 * z) - z * (t + 1)) / .01
    derivatives = 2 * (torch.sigmoid(2 * z) - (t + 1) / 2) * factors / .01
    masks = [t > .01, t < -.01, t.abs() <= .01]
    expected_loss = sum(float(losses[m].mean()) for m in masks) / 3
    expected_gradient = sum(derivatives[m].mean() for m in masks) / 3
    for chunk in [2, 3, 7]:
        with torch.no_grad():
            measured, _ = objective_pass(model, episodes, chunk, turn_group_counts=[2, 1, 1])
        assert measured == pytest.approx(expected_loss, abs=1e-5)
    model.weight.grad = None
    objective_pass(model, episodes, 7, backward=True, turn_group_counts=[2, 1, 1])
    torch.testing.assert_close(model.weight.grad, expected_gradient.float())


@pytest.mark.parametrize('counts', [[0, 2, 2], [1, 1], [1, 1, 1], [True, 1, 2]])
def test_balanced_objective_rejects_missing_or_inconsistent_original_groups(counts):
    episodes = [({}, {}, [], dict(target_valid=np.ones((4, 2), bool)))]
    with pytest.raises(ValueError, match='positive original training'):
        objective_pass(None, episodes, 64, turn_group_counts=counts)
