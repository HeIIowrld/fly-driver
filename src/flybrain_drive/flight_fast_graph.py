"""Capture unchanged rate updates after original sparse chemistry, on original intervals."""

import math

import numpy as np
import torch


def nominal_fast_step(raw, drive, state, coupling, alpha_fast, *, baseline):
    net = torch.sparse.mm(raw.chemical_matrix, state.T).T+drive
    return prepared_rate_step(raw, net, state, coupling, alpha_fast, baseline=baseline)


def prepared_rate_step(raw, net, state, coupling, alpha_fast, *, baseline):
    return prepared_interval_step(raw, net, state, coupling, alpha_fast,
                                  baseline=baseline, count=10)


def interval_parameters(raw, elapsed):
    nominal, tau = (raw.model.spec['dynamics'][key] for key in ('dt', 'tau'))
    if not math.isfinite(elapsed) or not 0 < elapsed <= nominal:
        raise ValueError('Original positive subinterval no longer than nominal required')
    count = int(math.ceil(elapsed/raw.maximum_microstep-1e-12))
    if not 1 <= count <= 10:
        raise ValueError('One to ten original microsteps required')
    nominal_alpha = float(np.float32(-np.expm1(-nominal/tau)))
    ratio = -np.expm1(-elapsed/tau)/nominal_alpha
    return count, -elapsed/count, ratio


def prepared_interval_step(raw, net, state, coupling, alpha_fast, *, baseline,
                           count, slow_ratio=None):
    target = torch.relu(torch.tanh(net))
    next_state = (1-raw.model.circuit.alpha)*state+raw.model.circuit.alpha*target
    if slow_ratio is not None:
        next_state = state+slow_ratio*(next_state-state)
    if baseline:
        return next_state
    fast, held = state[:, raw.positions], net[:, raw.positions]
    for _ in range(count):
        current = -coupling*(fast@raw.laplacian.T)
        fast = (1-alpha_fast)*fast+alpha_fast*torch.relu(torch.tanh(held+current))
    next_state[:, raw.positions] = fast
    return next_state


class GraphedFastHaltere:
    @torch.no_grad()
    def __init__(self, raw, *, coupling, fast_tau, graph_fractional=False):
        if (raw.model.spec['dynamics']['dt'] != .01 or raw.maximum_microstep != .001
                or raw.chemical_matrix.device.type != 'cuda'
                or not 0 <= coupling <= 1 or not .005 <= fast_tau <= .05):
            raise ValueError('Reviewed fixed10ms nominal,1ms microstep and GPU contract required')
        self.raw, self.coupling_value, self.tau_value = raw, coupling, fast_tau
        self.device = raw.chemical_matrix.device
        self.state = raw.model.initial_state(1)
        self.net = torch.zeros_like(self.state)
        self.coupling = torch.tensor(coupling, device=self.device).reshape(1, 1)
        self.alpha_fast = -torch.expm1(torch.tensor(-.001, device=self.device)
                                     /torch.tensor(fast_tau, device=self.device))
        self.fast_tau = torch.tensor(fast_tau, device=self.device).reshape(1, 1)
        self.interval_constants = torch.tensor([-.001, 1.], device=self.device)
        self.baseline = coupling == 0 and fast_tau == raw.model.spec['dynamics']['tau']
        self.mask_version = raw.model.circuit.state_mask._version
        if not bool(torch.all(raw.model.circuit.state_mask == 1)):
            raise ValueError('Unlesioned fixed state mask required for capture')
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                prepared_rate_step(raw, self.net, self.state, self.coupling,
                                   self.alpha_fast, baseline=self.baseline)
                if graph_fractional:
                    prepared_interval_step(raw, self.net, self.state, self.coupling,
                        -torch.expm1(self.interval_constants[0]/self.fast_tau),
                        baseline=self.baseline, count=10, slow_ratio=self.interval_constants[1])
        side.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.output = prepared_rate_step(raw, self.net, self.state, self.coupling,
                                            self.alpha_fast, baseline=self.baseline)
        self.fractional_graphs = {}
        if graph_fractional:
            # Private pools with shared explicit input tensors. Exactly one graph is
            # selected per original interval; no resampling or padding of neural time.
            for count in range(1, 11):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    output = prepared_interval_step(raw, self.net, self.state, self.coupling,
                        -torch.expm1(self.interval_constants[0]/self.fast_tau),
                        baseline=self.baseline, count=count,
                        slow_ratio=self.interval_constants[1])
                self.fractional_graphs[count] = (graph, output)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        self.nominal_steps = self.fractional_steps = 0
        self.fractional_graph_steps = 0

    @torch.no_grad()
    def advance(self, drive, state, elapsed):
        model = self.raw.model
        if (state.shape != self.state.shape or drive.shape != self.state.shape
                or state.dtype != self.state.dtype or state.device != self.device
                or drive.dtype != self.state.dtype or drive.device != self.device
                or (id(model.circuit.matrix), model.circuit.matrix._version)
                != self.raw.original_matrix_identity
                or model.circuit.state_mask._version != self.mask_version):
            raise ValueError('Same immutable matrix/mask and fixed single-stream tensors required')
        if elapsed != .01:
            if self.fractional_graphs:
                count, microdt, ratio = interval_parameters(self.raw, elapsed)
                if abs(ratio-1) <= 1e-7:
                    # Preserve the solver's near-nominal arithmetic branch exactly.
                    self.fractional_steps += 1
                    return self.raw.advance(drive, state, elapsed,
                        coupling=self.coupling_value, fast_tau=self.tau_value)
                net = torch.sparse.mm(self.raw.chemical_matrix, state.T).T+drive
                self.state.copy_(state)
                self.net.copy_(net)
                self.interval_constants.copy_(torch.tensor([microdt, ratio],
                    dtype=self.state.dtype, device='cpu'))
                graph, output = self.fractional_graphs[count]
                graph.replay()
                self.fractional_graph_steps += 1
                return output.clone()
            self.fractional_steps += 1
            return self.raw.advance(drive, state, elapsed,
                                    coupling=self.coupling_value, fast_tau=self.tau_value)
        # HIP sparse SpMM failed capture in t1266. Keep this original operation outside
        # the graph and capture only the unchanged dense rate-update recurrence.
        net = torch.sparse.mm(self.raw.chemical_matrix, state.T).T+drive
        self.state.copy_(state)
        self.net.copy_(net)
        self.graph.replay()
        self.nominal_steps += 1
        return self.output.clone()  # Previous actual states must not alias reusable graph output.
