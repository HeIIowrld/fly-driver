"""Multimodal recurrent fly-structure controller; engineering dynamics and vehicle readout."""

import numpy as np
import torch
from torch import nn

from .graph import validate_graph
from .local_retinal_circuit import LocalRetinalLinear
from .modality_ports import ModalityPortAdapter
from .sensory_policy import SensoryFeatureEncoder


def positive_gain(raw, parameterization='softplus'):
    if parameterization == 'softplus':
        return nn.functional.softplus(raw)
    if parameterization == 'log':
        return raw.exp()
    raise ValueError('Explicit positive gain parameterization required')


def encode_positive_gain(gain, parameterization='softplus'):
    gain = torch.as_tensor(gain)
    if not torch.isfinite(gain).all() or torch.any(gain <= 0):
        raise ValueError('Finite positive gain required')
    if parameterization == 'softplus':
        return gain + torch.log(-torch.expm1(-gain))
    if parameterization == 'log':
        return gain.log()
    raise ValueError('Explicit positive gain parameterization required')


class _FixedCSRMessage(torch.autograd.Function):
    """Fixed structural weights: explicit input adjoint with a cached CSR transpose."""

    @staticmethod
    def forward(ctx, state, matrix, transpose):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(matrix, state.T).T

    @staticmethod
    def backward(ctx, gradient):
        transpose, = ctx.saved_tensors
        return torch.sparse.mm(transpose, gradient.T).T, None, None


class _RectifiedTanhSurrogate(torch.autograd.Function):
    """Exact nonnegative forward rule; explicit engineering gradient through silent neurons."""

    @staticmethod
    def forward(ctx, value, negative_slope):
        activated = torch.tanh(value)
        ctx.save_for_backward(activated)
        ctx.negative_slope = negative_slope
        return torch.relu(activated)

    @staticmethod
    def backward(ctx, gradient):
        activated, = ctx.saved_tensors
        slope = torch.where(activated > 0, 1., ctx.negative_slope)
        return gradient * (1-activated.square()) * slope, None


class SignedRateCircuit(nn.Module):
    """Nonnegative bounded activity, fixed signed counts and full-source normalization.

    Transmitter sign and the common time constant are declared hypotheses. This
    surrogate has no spikes, delays, receptors, spontaneous drive or modulators.
    A zero sign suppresses outgoing effects without redistributing missing mass.
    """

    def __init__(self, graph, signs, incoming_totals, *, dt, tau, sparse_backend="coo",
                 activation_gradient='exact', surrogate_negative_slope=.05):
        super().__init__()
        validate_graph(graph)
        self.size = len(graph["ids"])
        if sparse_backend not in ("coo", "cached_csr_cpu", "cached_csr_gpu", "scipy_csr_cpu", "scipy_csr_compact_cpu", "numba_csr8_cpu"):
            raise ValueError("Explicit supported sparse message backend required")
        self.sparse_backend = sparse_backend
        if (activation_gradient not in ('exact', 'leaky_surrogate')
                or not np.isfinite(surrogate_negative_slope)
                or not 0 < surrogate_negative_slope <= 1):
            raise ValueError('Explicit supported activation gradient and slope required')
        self.activation_gradient = activation_gradient
        self.surrogate_negative_slope = surrogate_negative_slope
        self.register_buffer("_cached_matrix", None, persistent=False)
        self.register_buffer("_cached_transpose", None, persistent=False)
        self._cache_key = None
        self._scipy_cache_key = None
        self._scipy_matrix = None
        signs, totals = np.asarray(signs), np.asarray(incoming_totals)
        retained = np.bincount(graph["post"], weights=graph["weights"], minlength=self.size)
        if (signs.shape != (self.size,) or not np.isin(signs, [-1, 0, 1]).all()
                or totals.shape != (self.size,) or not np.isfinite(totals).all()
                or np.any(totals < retained) or not np.isfinite([dt, tau]).all()
                or min(dt, tau) <= 0):
            raise ValueError("Explicit signs, full incoming totals and positive timing required")
        weights = graph["weights"] * signs[graph["pre"]] / np.maximum(totals[graph["post"]], 1)
        matrix = torch.sparse_coo_tensor(np.array([graph["post"], graph["pre"]]),
                                        torch.tensor(weights, dtype=torch.float32),
                                        (self.size, self.size), check_invariants=True).coalesce()
        self.register_buffer("matrix", matrix)
        self.register_buffer("input_mask", torch.tensor(graph["roles"] == "input"))
        self.register_buffer("output_ports", torch.tensor(
            np.flatnonzero(graph["roles"] == "output")))
        self.register_buffer("state_mask", torch.ones(self.size))
        self.register_buffer("alpha", torch.tensor(-np.expm1(-dt / tau), dtype=torch.float32))

    def forward(self, drive, state):
        if (state.ndim != 2 or state.shape[1] != self.size or drive.shape != state.shape
                or not torch.isfinite(state).all() or torch.any((state < 0) | (state > 1))
                or not torch.isfinite(drive).all() or torch.any(drive[:, ~self.input_mask])):
            raise ValueError("Bounded persistent activity and input-port-only drive required")
        return self._advance_tensor(drive, state)

    def _advance_tensor(self, drive, state):
        """Same dynamics; caller must validate a sealed training chunk at both boundaries."""
        if self.sparse_backend in ("scipy_csr_cpu", "scipy_csr_compact_cpu", "numba_csr8_cpu"):
            if (torch.is_grad_enabled() or state.requires_grad or drive.requires_grad
                    or state.device.type != 'cpu' or drive.device.type != 'cpu'
                    or state.dtype != torch.float32 or self.matrix.dtype != torch.float32
                    or self.matrix.device.type != 'cpu' or self.matrix.requires_grad):
                raise ValueError('SciPy CSR is fixed float32 CPU inference under no_grad only')
            key = (id(self.matrix), self.matrix._version, self.matrix.device, self.matrix.dtype, self.sparse_backend)
            if key != self._scipy_cache_key:
                import scipy.sparse

                matrix = self.matrix.coalesce().to_sparse_csr()
                self._scipy_matrix = scipy.sparse.csr_matrix(
                    (matrix.values().numpy().copy(), matrix.col_indices().numpy().copy(),
                     matrix.crow_indices().numpy().copy()), shape=tuple(matrix.shape))
                if self.sparse_backend in ("scipy_csr_compact_cpu", "numba_csr8_cpu"):
                    self._scipy_matrix.eliminate_zeros()
                self._scipy_cache_key = key
            if self.sparse_backend == "numba_csr8_cpu":
                from .parallel_csr_cpu import fixed_csr_matmul
                message = torch.from_numpy(fixed_csr_matmul(self._scipy_matrix, state.numpy()))
            else:
                message = torch.from_numpy((self._scipy_matrix @ state.numpy().T).T.copy())
        elif self.sparse_backend in ("cached_csr_cpu", "cached_csr_gpu"):
            expected_device = "cpu" if self.sparse_backend == "cached_csr_cpu" else "cuda"
            if state.device.type != expected_device or self.matrix.requires_grad:
                raise ValueError("Cached CSR requires fixed weights on its declared device")
            # Tensor mutation version catches load_state_dict/in-place edits; identity catches cuts.
            key = (id(self.matrix), self.matrix._version, self.matrix.device, self.matrix.dtype)
            if key != self._cache_key:
                matrix = self.matrix.coalesce()
                self._cached_matrix = matrix.to_sparse_csr()
                self._cached_transpose = matrix.transpose(0, 1).coalesce().to_sparse_csr()
                self._cache_key = key
            message = _FixedCSRMessage.apply(state, self._cached_matrix, self._cached_transpose)
        else:
            message = torch.sparse.mm(self.matrix, state.T).T
        net = message + drive
        activated = torch.relu(torch.tanh(net)) if self.activation_gradient == 'exact' else \
            _RectifiedTanhSurrogate.apply(net, self.surrogate_negative_slope)
        state = ((1 - self.alpha) * state + self.alpha * activated) * self.state_mask
        return state[:, self.output_ports], state


class VehicleReadout(nn.Module):
    """Explicit output-only action readout, no bias or direct sensory/motor residual.

    Steering is a declared differential orientation. Pedals use positive weighted
    output activity. Legacy mode attenuates throttle with brake; independent mode
    preserves separate pedal requests. This is a vehicle codec hypothesis,
    not a measured fly reflex. Actions are requests, never applied-control receipts.
    """

    def __init__(self, graph, groups, steering_signs, *, pedal_mode='brake_attenuated',
                 gain_parameterization='softplus'):
        super().__init__()
        if pedal_mode not in ('brake_attenuated', 'independent'):
            raise ValueError('Explicit supported pedal mode required')
        self.pedal_mode = pedal_mode
        if gain_parameterization not in ('softplus', 'log'):
            raise ValueError('Explicit positive gain parameterization required')
        self.gain_parameterization = gain_parameterization
        if set(groups) != {"steering", "forward", "halt"}:
            raise ValueError("Three explicit disjoint output groups required")
        output_ids = [str(b) for b, role in zip(graph["ids"], graph["roles"], strict=True)
                      if role == "output"]
        flat = [str(b) for group in groups.values() for b in group]
        if (len(flat) != len(set(flat)) or set(flat) != set(output_ids)
                or any(not len(group) for group in groups.values())
                or len(groups["steering"]) != 2 or len(steering_signs) != 2
                or set(steering_signs) != {-1, 1}):
            raise ValueError("Cover exactly all outputs and specify steering signs")
        for name, bodies in groups.items():
            self.register_buffer(name + "_ports", torch.tensor([output_ids.index(str(b))
                                                                 for b in bodies]))
        self.register_buffer("steering_signs", torch.tensor(steering_signs, dtype=torch.float32))
        initial = 0.54132485 if gain_parameterization == 'softplus' else 0.
        self.steering_log_gain = nn.Parameter(torch.tensor(initial))
        self.pedal_log_gain = nn.Parameter(torch.full((2,), initial))
        self.forward_weights = nn.Parameter(torch.zeros(len(groups["forward"])))
        self.halt_weights = nn.Parameter(torch.zeros(len(groups["halt"])))
        self.output_size = len(output_ids)

    def forward(self, outputs):
        if (outputs.ndim != 2 or outputs.shape[1] != self.output_size
                or not torch.isfinite(outputs).all() or torch.any((outputs < 0) | (outputs > 1))):
            raise ValueError("Readout accepts only bounded declared graph output activity")
        return self._decode_tensor(outputs)

    def _decode_tensor(self, outputs):
        """Tensor-only decoder used after chunk boundary validation during offline training."""
        steer = torch.tanh(positive_gain(self.steering_log_gain, self.gain_parameterization)
                           * (outputs[:, self.steering_ports] @ self.steering_signs))
        forward = outputs[:, self.forward_ports] @ self.forward_weights.softmax(0)
        halt = outputs[:, self.halt_ports] @ self.halt_weights.softmax(0)
        gains = positive_gain(self.pedal_log_gain, self.gain_parameterization)
        brake = -torch.expm1(-gains[1] * halt)
        throttle = -torch.expm1(-gains[0] * forward)
        if self.pedal_mode == 'brake_attenuated':
            throttle = throttle * (1 - brake)
        return torch.stack([steer, throttle, brake], dim=1)


class VehicleCircuitController(nn.Module):
    """Local eyes and declared organ channels must traverse the recurrent graph."""

    def __init__(self, graph, visual_ids, organ_groups, directions, available, organ_size,
                 signs, incoming_totals, output_groups, steering_signs, *,
                 organ_calibration_sha256, eye_map_hashes,
                 neighbors=12, seed=790101, dt=.01, tau=.05):
        super().__init__()
        if (not isinstance(organ_calibration_sha256, str)
                or len(organ_calibration_sha256) != 64
                or any(c not in "0123456789abcdef" for c in organ_calibration_sha256)):
            raise ValueError("Explicit organ calibration SHA256 required")
        self.organ_calibration_sha256 = organ_calibration_sha256
        if (set(eye_map_hashes) != {"left", "right"}
                or any(not isinstance(h, str) or len(h) != 64
                       or any(c not in "0123456789abcdef" for c in h)
                       for h in eye_map_hashes.values())):
            raise ValueError("Explicit paired eye-map hashes required")
        self.eye_map_hashes = dict(eye_map_hashes)
        local = LocalRetinalLinear(directions, available, len(visual_ids), neighbors, seed)
        width = np.asarray(directions).size
        self.adapter = ModalityPortAdapter(graph, visual_ids, organ_groups, organ_size, width,
                                          visual_adapter=local)
        self.circuit = SignedRateCircuit(graph, signs, incoming_totals, dt=dt, tau=tau)
        self.readout = VehicleReadout(graph, output_groups, steering_signs)
        self.register_buffer("directions", torch.tensor(directions, dtype=torch.float32))
        self.register_buffer("feature_mean", torch.zeros(width))
        self.register_buffer("feature_scale", torch.ones(width))

    def initial_state(self, batch=1):
        return self.feature_mean.new_zeros((batch, self.circuit.size))

    def prepare_drive(self, observation):
        if set(observation) != SensoryFeatureEncoder.fields:
            raise ValueError("Only declared actor fields are accepted")
        rgb, valid = observation["ray_values"], observation["ray_valid"]
        if (rgb.ndim != 4 or rgb.shape[1:] != self.directions.shape
                or valid.shape != rgb.shape[:-1] or not torch.isfinite(rgb).all()
                or torch.any((rgb < 0) | (rgb > 1)) or not ((valid == 0) | (valid == 1)).all()
                or observation["ray_directions"].shape != rgb.shape
                or not torch.equal(observation["ray_directions"], self.directions.expand_as(rgb))
                or not torch.isfinite(self.feature_scale).all()
                or torch.any(self.feature_scale <= 0)):
            raise ValueError("Retinal values, masks, geometry or normalization violate contract")
        mask = valid[..., None].expand_as(rgb).flatten(1)
        features = ((rgb.flatten(1) - self.feature_mean) / self.feature_scale) * mask
        return self.adapter(features, valid.flatten(1).any(1, keepdim=True).float(),
                            observation["organs"], observation["organ_valid"])

    def advance(self, drive, state):
        outputs, state = self.circuit(drive, state)
        return self.readout(outputs), state

    def forward(self, observation, state):
        return self.advance(self.prepare_drive(observation), state)


class VehicleCircuitRunner:
    """Persistent inference on validated ActorObservation, without any actuator sender."""

    def __init__(self, model, packer):
        if (model.adapter.organ_size != packer.organ_size
                or model.organ_calibration_sha256 != packer.organs.calibration_sha256
                or model.eye_map_hashes != {side: eye.sha256 for side, eye in packer.eyes.items()}):
            raise ValueError("Model and packer sensory calibration differ")
        self.model, self.packer = model, packer
        self.model.eval()
        self.reset()

    def reset(self):
        self.episode, self.state = None, None
        self.packer.episode = None
        self.packer.exposures = {}
        self.packer.organs.reset()

    def step(self, observation):
        packed = self.packer.pack(observation)
        if observation.episode_id != self.episode:
            self.state = self.model.initial_state()
            self.episode = observation.episode_id
        tensors = {key: torch.tensor(value, device=self.model.feature_mean.device)[None]
                   for key, value in packed.items()}
        with torch.no_grad():
            requests, self.state = self.model(tensors, self.state)
        return requests[0].cpu().numpy(), packed
