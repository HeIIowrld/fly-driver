"""Synthetic protocol tests; these fixtures are not measured visual neurons."""

from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.flyvis_frozen_tail import FrozenVisualTail, install_frozen_visual_tail


class PPNeuronIGRSynapses:
    pass


class Stimulus:
    def zero(self, samples, frames):
        self.buffer = torch.zeros(samples, frames, 721)

    def add_input(self, value):
        self.buffer += value.squeeze(2)

    def __call__(self):
        return self.buffer


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.dynamics, self.stimulus = PPNeuronIGRSynapses(), Stimulus()
        self._state_hooks, self.parameter_builds, self.steps = (), 0, []
        self.eval()

    def clamp(self):
        pass

    def _param_api(self):
        self.parameter_builds += 1
        return self.weight.clone()

    def _next_state(self, params, state, current, dt):
        self.steps.append(dt)
        return state+current*params*dt

    def simulate(self):
        raise AssertionError('Original simulator is not a silent fallback')


def test_every_original_step_is_executed_while_only_last_state_is_retained():
    network = Fixture()
    backend = FrozenVisualTail(network)
    held = torch.ones(4, 1, 1, 721).expand(-1, 8, -1, -1)
    output = backend.simulate(held, .005, torch.zeros(4, 721), as_states=True)
    assert len(output) == 1 and network.steps == [.005]*8 and network.parameter_builds == 1
    torch.testing.assert_close(output[-1], torch.full((4, 721), .04))


def test_nonheld_movie_and_changed_parameters_are_rejected_without_silent_repair():
    network = Fixture()
    backend = FrozenVisualTail(network)
    with pytest.raises(ValueError, match='constant'):
        backend.simulate(torch.ones(4, 8, 1, 721), .005,
                         torch.zeros(4, 721), as_states=True)
    with torch.no_grad():
        network.weight.add_(1.)
    with pytest.raises(ValueError, match='frozen'):
        backend.simulate(torch.ones(4, 1, 1, 721), .005,
                         torch.zeros(4, 721), as_states=True)


def test_backend_cannot_replace_an_existing_neural_episode():
    with pytest.raises(ValueError, match='inside an observed'):
        install_frozen_visual_tail(SimpleNamespace(steps=1))
