"""Optional captured held-input rate integrator with original 5ms/fractional steps."""

import math

import numpy as np
import torch

from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow


class GraphedHeldBodyRate:
    """Capture pointwise rate step only; original sparse multiplication remains eager."""

    @torch.no_grad()
    def __init__(self, joint):
        if type(joint) is not ExperimentalVisualBodyAirflow:
            raise ValueError('Exact joint sensory circuit required')
        circuit = joint.circuit.backbone.circuit
        spec = joint.circuit.backbone.spec['dynamics']
        if (circuit.matrix.device.type != 'cuda' or circuit.matrix.dtype != torch.float32
                or circuit.sparse_backend != 'cached_csr_gpu' or circuit.activation_gradient != 'exact'
                or not .005 <= spec['dt'] or spec['tau'] <= 0
                or not torch.all(circuit.state_mask == 1)):
            raise ValueError('Unlesioned original CUDA rate circuit and >=5ms nominal step required')
        joint.circuit.backbone.assert_body_boundary_protected()
        self.joint, self.circuit = joint, circuit
        self.nominal, self.tau = float(spec['dt']), float(spec['tau'])
        self.matrix = circuit.matrix.coalesce().to_sparse_csr()
        self.identity = (id(circuit.matrix), circuit.matrix._version,
                         circuit.state_mask._version, circuit.alpha._version)
        self.state = joint.initial_state(1)
        self.drive = torch.zeros_like(self.state)
        self.net = torch.zeros_like(self.state)
        self.ratio = torch.tensor(self._ratio(.005), dtype=torch.float32, device='cuda')

        def step(state):
            activated = torch.relu(torch.tanh(self.net))
            nominal_next = ((1-circuit.alpha)*state+circuit.alpha*activated)*circuit.state_mask
            return state+self.ratio*(nominal_next-state)

        side = torch.cuda.Stream(device='cuda')
        side.wait_stream(torch.cuda.current_stream('cuda'))
        with torch.cuda.stream(side):
            for _ in range(3):
                step(self.state)
        side.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.output = step(self.state)
        torch.cuda.current_stream('cuda').wait_stream(side)
        torch.cuda.synchronize()

    def _ratio(self, elapsed):
        alpha = float(np.float32(-np.expm1(-self.nominal/self.tau)))
        return -np.expm1(-elapsed/self.tau)/alpha

    @torch.no_grad()
    def advance(self, drive, state, intervals):
        """Apply every explicit original interval; no padding or numerical reset."""
        current_identity = (id(self.circuit.matrix), self.circuit.matrix._version,
                            self.circuit.state_mask._version, self.circuit.alpha._version)
        if (current_identity != self.identity or not isinstance(intervals, list) or not intervals
                or any(not isinstance(dt, (int, float)) or isinstance(dt, bool)
                       or not math.isfinite(dt) or not 0 < dt <= .005 for dt in intervals)
                or state.shape != self.state.shape or drive.shape != self.drive.shape
                or state.device != self.state.device or drive.device != self.drive.device
                or state.dtype != self.state.dtype or drive.dtype != self.drive.dtype
                or not torch.isfinite(state).all() or not torch.isfinite(drive).all()
                or torch.any((state < 0) | (state > 1))):
            raise ValueError('Original bounded state, drive, immutable graph and intervals required')
        self.joint.circuit.backbone.assert_body_boundary_protected()
        result = state
        for dt in intervals:
            # rocsparse CSR multiplication is not graph-capturable on the
            # tested runtime. Its original eager operator is retained per step.
            net = torch.sparse.mm(self.matrix, result.T).T+drive
            self.state.copy_(result)
            self.net.copy_(net)
            self.ratio.fill_(self._ratio(dt))
            self.graph.replay()
            result = self.output.clone()
        if not torch.isfinite(result).all() or torch.any((result < 0) | (result > 1)):
            raise ValueError('Captured rate output must remain finite and bounded')
        return result
