"""Synthetic protocol tests; measured-neuron parity is a separate full-history GPU replay."""

import pytest
import torch

from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail, PreparedDynamicStep


def test_only_the_owned_dynamic_scalar_is_substituted_and_its_value_is_not_frozen():
    value = torch.tensor(.00317)
    with PreparedDynamicStep(value):
        assert torch.tensor(value) is value
        assert torch.tensor(.002).item() == pytest.approx(.002)
        value.fill_(.00123)
        assert torch.tensor(value).item() == pytest.approx(.00123)
        with pytest.raises(ValueError):
            torch.tensor(value, dtype=torch.float64)


def test_fractional_dt_changes_are_preserved_and_old_outputs_are_not_overwritten():
    class Namespace(dict):
        __getattr__ = dict.__getitem__

    class Stimulus:
        def zero(self, *args):
            self.value = torch.zeros(4, 1, 721)

        def add_input(self, value):
            self.value += value.squeeze(2)

        def __call__(self):
            return self.value

    network = torch.nn.Module().eval()
    network.stimulus, network._state_hooks = Stimulus(), ()
    network._state_api = lambda state: state
    backend = object.__new__(FractionalVisualTail)
    backend.network, backend.device = network, torch.device('cpu')
    backend.versions, backend.AutoDeref = (), Namespace
    backend.static_activity = torch.zeros(4, 721)
    backend.static_current = torch.zeros(4, 721)
    backend.fractional_dt = torch.tensor(.00317)
    backend.fractional_output = Namespace(nodes=Namespace(activity=torch.zeros(4, 721)))
    backend.graphed_launches = backend.fractional_steps = backend.steps = backend.calls = 0

    def replay():
        backend.fractional_output.nodes.activity.copy_(
            backend.static_activity + backend.static_current*backend.fractional_dt)

    # Instance function, expressly not a captured GPU graph or measured neuronal dynamics.
    backend.fractional_graph = type('GraphFixture', (), {'replay': staticmethod(replay)})()
    initial = Namespace(nodes=Namespace(activity=torch.ones(4, 721)), edges=Namespace())
    held = torch.linspace(0, 1, 721).expand(4, 1, 1, -1)
    retained = []
    for dt in (.00317, .00000123, .004999999, .0012):
        output = backend.simulate(held, dt, initial, as_states=True)[0]
        expected = initial.nodes.activity + held[:, 0, 0]*dt
        torch.testing.assert_close(output.nodes.activity, expected, atol=0, rtol=0)
        retained.append((output, output.nodes.activity.clone()))
    for output, saved in retained:
        torch.testing.assert_close(output.nodes.activity, saved, atol=0, rtol=0)
    for dt in (0., -.001, .006):
        with pytest.raises(ValueError, match='fractional'):
            backend.simulate(held, dt, initial, as_states=True)
    assert backend.steps == backend.fractional_steps == backend.graphed_launches == 4
