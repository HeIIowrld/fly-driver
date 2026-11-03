"""One private captured graph for each original1..16-step nominal held-input batch."""

import torch

from .flyvis_graphed_tail import GraphedVisualTail, PreparedNominalConstant


class FullBatchVisualTail(GraphedVisualTail):
    """Same author step function, constants and state copies; no binary chunk launches."""

    @torch.no_grad()
    def __init__(self, network, initial_state):
        super().__init__(network, initial_state)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        for count in range(1, 17):
            if count in self.graphs:
                continue
            graph = torch.cuda.CUDAGraph()
            with (torch.cuda.graph(graph, stream=side), torch.device(self.device),
                  PreparedNominalConstant(self.nominal_dt)):
                state = self._input_state()
                for _ in range(count):
                    state = network._next_state(self.params, state, self.static_current, .005)
            self.graphs[count] = (graph, state)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def simulate(self, movie_input, dt, initial_state, as_states=False, as_layer_activity=False):
        if dt != .005:
            return super().simulate(movie_input, dt, initial_state, as_states, as_layer_activity)
        if (not as_states or as_layer_activity or initial_state is None
                or movie_input.ndim != 4 or movie_input.shape[0] != 4
                or movie_input.shape[2:] != (1, 721) or not 1 <= movie_input.shape[1] <= 16
                or movie_input.shape[1] > 1 and movie_input.stride(1) != 0
                or movie_input.device != self.device or movie_input.dtype != torch.float32
                or self.network.training or any(p.requires_grad for p in self.network.parameters())
                or self.network._state_hooks or self._versions() != self.versions
                or set(initial_state.nodes) != {'activity'} or initial_state.edges):
            raise ValueError('Same frozen graph, original held inputs and explicit state required')
        count = movie_input.shape[1]
        with torch.device(self.device):
            self.network.stimulus.zero(4, 1)
            self.network.stimulus.add_input(movie_input[:, :1])
            self.static_current.copy_(self.network.stimulus()[:, 0])
            self.static_activity.copy_(initial_state.nodes.activity)
            graph, output = self.graphs[count]
            graph.replay()
            state = self.network._state_api(self.AutoDeref(
                nodes=self.AutoDeref(activity=output.nodes.activity.clone()),
                edges=self.AutoDeref()))
        self.graphed_launches += 1
        self.graphed_steps += count
        self.steps += count
        self.calls += 1
        return [state]


def install_fullbatch_visual_tail(stream):
    if stream.steps or stream.order.clock is not None or hasattr(stream, 'frozen_visual_tail'):
        raise ValueError('One explicit backend prepared before the original observed episode')
    stream.frozen_visual_tail = FullBatchVisualTail(stream.network, stream.visual_state)
    stream.network.simulate = stream.frozen_visual_tail.simulate
    return stream
