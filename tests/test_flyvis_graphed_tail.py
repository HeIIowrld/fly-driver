"""Protocol safeguards only; GPU and measured-neuron parity require the recorded replay."""

from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.flyvis_fullbatch_tail import FullBatchVisualTail, install_fullbatch_visual_tail
from flybrain_drive.flyvis_graphed_tail import (
    GraphedVisualTail,
    PreparedNominalConstant,
    install_graphed_visual_tail,
)


def test_prepared_scalar_cannot_silently_replace_a_fractional_integration_interval():
    nominal = torch.tensor(.005)
    with PreparedNominalConstant(nominal):
        assert torch.tensor(.005) is nominal
        with pytest.raises(ValueError, match='nominal'):
            torch.tensor(.00317)
        with pytest.raises(ValueError, match='nominal'):
            torch.tensor(.005, dtype=torch.float64)
        torch.testing.assert_close(torch.ones(3)*2, torch.full((3,), 2.))
    assert torch.tensor(.00317).item() == pytest.approx(.00317)


@pytest.mark.parametrize('install', [install_graphed_visual_tail, install_fullbatch_visual_tail])
def test_graph_installation_cannot_reset_or_replace_an_observed_episode(install):
    with pytest.raises(ValueError, match='before'):
        install(SimpleNamespace(steps=1))
    with pytest.raises(ValueError, match='before'):
        install(SimpleNamespace(steps=0, order=SimpleNamespace(clock=1.)))


@pytest.mark.parametrize('backend_type, counts, maximum_launches', [
    (GraphedVisualTail, (1, 2, 4, 8), 4),
    (FullBatchVisualTail, range(1, 17), 1),
])
def test_chunk_replays_preserve_every_step_and_do_not_overwrite_returned_state(
        backend_type, counts, maximum_launches):
    # Synthetic additive recurrence, expressly not a measured-neuron fixture.
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
    backend = object.__new__(backend_type)
    backend.network, backend.device = network, torch.device('cpu')
    backend.versions, backend.AutoDeref = (), Namespace
    backend.static_activity = torch.zeros(4, 721)
    backend.static_current = torch.zeros(4, 721)
    backend.calls = backend.steps = backend.graphed_steps = backend.graphed_launches = 0
    backend.fractional_steps = 0
    launches = []

    def graph(count):
        state = Namespace(nodes=Namespace(activity=torch.zeros(4, 721)), edges=Namespace())

        def replay():
            value = backend.static_activity.clone()
            for _ in range(count):
                value = value + backend.static_current*.005
            state.nodes.activity.copy_(value)
            launches.append(count)

        return SimpleNamespace(replay=replay), state

    backend.graphs = {count: graph(count) for count in counts}
    initial = Namespace(nodes=Namespace(activity=torch.ones(4, 721)), edges=Namespace())
    for count in range(1, 17):
        launches.clear()
        held = torch.ones(4, 1, 1, 721).expand(-1, count, -1, -1)
        result = backend.simulate(held, .005, initial, as_states=True)[0]
        expected = initial.nodes.activity.clone()
        for _ in range(count):
            expected += .005
        torch.testing.assert_close(result.nodes.activity, expected, rtol=0, atol=0)
        assert sum(launches) == count and len(launches) <= maximum_launches
        retained = result.nodes.activity.clone()
        backend.simulate(held*0 if count == 1 else held, .005, initial, as_states=True)
        torch.testing.assert_close(result.nodes.activity, retained, rtol=0, atol=0)
