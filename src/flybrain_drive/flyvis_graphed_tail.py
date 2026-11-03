"""Capture the unchanged author 5ms step; fractional steps remain original eager calls."""

import torch
from torch.overrides import TorchFunctionMode

from .flyvis_frozen_tail import FrozenVisualTail


class PreparedNominalConstant(TorchFunctionMode):
    """Replace only the upstream literal dt allocation with the identical prepared scalar."""

    def __init__(self, value):
        self.value = value

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.tensor and len(args) == 1 and type(args[0]) is float:
            if args[0] != .005 or kwargs:
                raise ValueError('Only the original nominal dt scalar may be prepared')
            return self.value
        return func(*args, **(kwargs or {}))


class GraphedVisualTail(FrozenVisualTail):
    """Independent static graph buffers; returned states never alias reusable graph outputs."""

    @torch.no_grad()
    def __init__(self, network, initial_state):
        super().__init__(network)
        if (set(initial_state.nodes) != {'activity'} or initial_state.edges
                or initial_state.nodes.activity.shape != (4, 45669)
                or self.device.type != 'cuda'):
            raise ValueError('Exact published four-camera activity-only CUDA state required')
        from flyvis.utils.tensor_utils import AutoDeref

        self.AutoDeref = AutoDeref
        self.static_activity = initial_state.nodes.activity.clone()
        self.static_current = torch.zeros_like(self.static_activity)
        self.nominal_dt = torch.tensor(.005, device=self.device, dtype=torch.float32)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with (torch.cuda.stream(side), torch.device(self.device),
              PreparedNominalConstant(self.nominal_dt)):
            for _ in range(3):
                network._next_state(self.params, self._input_state(), self.static_current, .005)
        side.synchronize()
        self.graphs = {}
        # Four private pools, no shared-pool ordering assumption. Binary chunks preserve
        # every step while requiring at most four launches for any admitted held batch.
        for count in (1, 2, 4, 8):
            graph = torch.cuda.CUDAGraph()
            with (torch.cuda.graph(graph, stream=side), torch.device(self.device),
                  PreparedNominalConstant(self.nominal_dt)):
                state = self._input_state()
                for _ in range(count):
                    state = network._next_state(self.params, state, self.static_current, .005)
            self.graphs[count] = (graph, state)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.graphed_steps = self.fractional_steps = self.graphed_launches = 0

    def _input_state(self):
        # RefTensor gathers cache their first result. Construct fresh lazy references so
        # source/target gathers are captured as operations, not frozen warmup activations.
        return self.network._state_api(self.AutoDeref(
            nodes=self.AutoDeref(activity=self.static_activity), edges=self.AutoDeref()))

    @torch.no_grad()
    def simulate(self, movie_input, dt, initial_state, as_states=False, as_layer_activity=False):
        if (not as_states or as_layer_activity or initial_state is None
                or not 0 < dt <= .005 or movie_input.ndim != 4
                or movie_input.shape[0] != 4 or movie_input.shape[2:] != (1, 721)
                or not 1 <= movie_input.shape[1] <= 16
                or movie_input.shape[1] > 1 and movie_input.stride(1) != 0
                or movie_input.device != self.device or movie_input.dtype != torch.float32
                or self.network.training or any(p.requires_grad for p in self.network.parameters())
                or self.network._state_hooks or self._versions() != self.versions
                or set(initial_state.nodes) != {'activity'} or initial_state.edges):
            raise ValueError('Same frozen graph, original held inputs and explicit state required')
        with torch.device(self.device):
            self.network.stimulus.zero(4, 1)
            self.network.stimulus.add_input(movie_input[:, :1])
            current = self.network.stimulus()[:, 0]
            if dt == .005:
                self.static_current.copy_(current)
                self.static_activity.copy_(initial_state.nodes.activity)
                remaining = movie_input.shape[1]
                for count in (8, 4, 2, 1):
                    while remaining >= count:
                        graph, output = self.graphs[count]
                        graph.replay()
                        remaining -= count
                        self.graphed_launches += 1
                        if remaining:
                            self.static_activity.copy_(output.nodes.activity)
                state = self.network._state_api(self.AutoDeref(
                    nodes=self.AutoDeref(activity=output.nodes.activity.clone()),
                    edges=self.AutoDeref()))
                self.graphed_steps += movie_input.shape[1]
            else:
                state = initial_state
                for _ in range(movie_input.shape[1]):
                    state = self.network._next_state(self.params, state, current, dt)
                self.fractional_steps += movie_input.shape[1]
        self.calls += 1
        self.steps += movie_input.shape[1]
        return [state]


def install_graphed_visual_tail(stream):
    if stream.steps or stream.order.clock is not None or hasattr(stream, 'frozen_visual_tail'):
        raise ValueError('One explicit backend prepared before the original observed episode')
    stream.frozen_visual_tail = GraphedVisualTail(stream.network, stream.visual_state)
    stream.network.simulate = stream.frozen_visual_tail.simulate
    return stream
