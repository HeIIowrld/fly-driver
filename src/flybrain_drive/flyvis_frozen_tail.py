"""Fixed-parameter author steps with one held input and only the final state retained."""

import torch


class FrozenVisualTail:
    """Same upstream _next_state, edge summation, dtype and dt; no state/time approximation."""

    def __init__(self, network):
        if (network.training or any(p.requires_grad for p in network.parameters())
                or network._state_hooks
                or type(network.dynamics).__name__ != 'PPNeuronIGRSynapses'):
            raise ValueError('Frozen hook-free published point-neuron network required')
        self.network = network
        self.device = next(network.parameters()).device
        self.original = network.simulate
        self.versions = self._versions()
        network.clamp()  # Exactly the upstream no-op for frozen parameters.
        self.params = network._param_api()
        self.calls = self.steps = 0

    def _versions(self):
        return tuple((name, id(value), value._version) for name, value in
                     [*self.network.named_parameters(), *self.network.named_buffers()])

    @torch.no_grad()
    def simulate(self, movie_input, dt, initial_state, as_states=False, as_layer_activity=False):
        if (not as_states or as_layer_activity or initial_state is None
                or not 0 < dt <= .005 or movie_input.ndim != 4
                or movie_input.shape[0] != 4 or movie_input.shape[2:] != (1, 721)
                or not 1 <= movie_input.shape[1] <= 16
                or movie_input.shape[1] > 1 and movie_input.stride(1) != 0
                or movie_input.device != self.device or movie_input.dtype != torch.float32
                or self.network.training or any(p.requires_grad for p in self.network.parameters())
                or self.network._state_hooks or self._versions() != self.versions):
            raise ValueError('Same frozen network and constant4-camera held input required')
        with torch.device(self.device):
            # Reuse the original stimulus transduction/indexing, for one identical held frame.
            self.network.stimulus.zero(4, 1)
            self.network.stimulus.add_input(movie_input[:, :1])
            current = self.network.stimulus()[:, 0]
            state = initial_state
            for _ in range(movie_input.shape[1]):
                state = self.network._next_state(self.params, state, current, dt)
        self.calls += 1
        self.steps += movie_input.shape[1]
        # Owned inference callers explicitly consume only[-1]; this is not the full public API.
        return [state]


def install_frozen_visual_tail(stream):
    if stream.steps or stream.order.clock is not None:
        raise ValueError('No visual backend replacement inside an observed neural episode')
    if hasattr(stream, 'frozen_visual_tail'):
        raise ValueError('One explicit visual backend per original episode')
    stream.frozen_visual_tail = FrozenVisualTail(stream.network)
    stream.network.simulate = stream.frozen_visual_tail.simulate
    return stream
