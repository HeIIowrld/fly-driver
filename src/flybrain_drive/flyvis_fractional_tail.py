"""Original fractional dt as a dynamic float32 scalar, with unchanged author equations."""

import torch
from torch.overrides import TorchFunctionMode

from .flyvis_graphed_tail import GraphedVisualTail


class PreparedDynamicStep(TorchFunctionMode):
    """Supply only the exact captured dt tensor to the author's torch.tensor(dt).float()."""

    def __init__(self, value):
        self.value = value

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.tensor and len(args) == 1 and args[0] is self.value:
            if kwargs:
                raise ValueError('Only the original dt scalar allocation may be replaced')
            return self.value
        return func(*args, **(kwargs or {}))


class FractionalVisualTail(GraphedVisualTail):
    """Nominal binary graphs unchanged; one original fractional step with dynamic dt input."""

    @torch.no_grad()
    def __init__(self, network, initial_state, *, batch_size=4, extra_chunks=()):
        super().__init__(network, initial_state, batch_size=batch_size, extra_chunks=extra_chunks)
        self.fractional_dt = torch.tensor(.00317, device=self.device, dtype=torch.float32)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with (torch.cuda.stream(side), torch.device(self.device),
              PreparedDynamicStep(self.fractional_dt)):
            for _ in range(3):
                network._next_state(self.params, self._input_state(), self.static_current,
                                    self.fractional_dt)
        side.synchronize()
        self.fractional_graph = torch.cuda.CUDAGraph()
        with (torch.cuda.graph(self.fractional_graph, stream=side), torch.device(self.device),
              PreparedDynamicStep(self.fractional_dt)):
            self.fractional_output = network._next_state(self.params, self._input_state(),
                                                        self.static_current, self.fractional_dt)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def simulate(self, movie_input, dt, initial_state, as_states=False, as_layer_activity=False):
        if dt == .005:
            return super().simulate(movie_input, dt, initial_state, as_states, as_layer_activity)
        if (not as_states or as_layer_activity or initial_state is None or not 0 < dt < .005
                or movie_input.shape != (self.batch_size, 1, 1, 721)
                or movie_input.device != self.device or movie_input.dtype != torch.float32
                or self.network.training or any(p.requires_grad for p in self.network.parameters())
                or self.network._state_hooks or self._versions() != self.versions
                or set(initial_state.nodes) != {'activity'} or initial_state.edges):
            raise ValueError('One exact fractional held step in the same frozen network required')
        with torch.device(self.device):
            self.network.stimulus.zero(self.batch_size, 1)
            self.network.stimulus.add_input(movie_input)
            self.static_current.copy_(self.network.stimulus()[:, 0])
            self.static_activity.copy_(initial_state.nodes.activity)
            # Both upstream dt allocations/multiplications already operate in float32.
            # Fill with this original fractional dt; never round to a nominal time step.
            self.fractional_dt.fill_(dt)
            self.fractional_graph.replay()
            state = self.network._state_api(self.AutoDeref(
                nodes=self.AutoDeref(activity=self.fractional_output.nodes.activity.clone()),
                edges=self.AutoDeref()))
        self.graphed_launches += 1
        self.fractional_steps += 1
        self.steps += 1
        self.calls += 1
        return [state]


def install_fractional_visual_tail(stream):
    if stream.steps or stream.order.clock is not None or hasattr(stream, 'frozen_visual_tail'):
        raise ValueError('One explicit backend prepared before the original observed episode')
    stream.frozen_visual_tail = FractionalVisualTail(stream.network, stream.visual_state)
    stream.network.simulate = stream.frozen_visual_tail.simulate
    return stream
