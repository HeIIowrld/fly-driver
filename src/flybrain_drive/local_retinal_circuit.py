"""Local retinal features into MeTu ports; engineering adapter, not anatomical registration."""

import numpy as np
import torch
from torch import nn

from .graph import validate_graph
from .modality_ports import ModalityPortAdapter
from .sensory_policy import SensoryFeatureEncoder


class LocalRetinalLinear(nn.Module):
    """Each neuron receives a small neighborhood in one eye, without an action bottleneck."""

    def __init__(self, directions, available, ports, neighbors, seed):
        super().__init__()
        directions, available = np.asarray(directions), np.asarray(available)
        if (directions.ndim != 3 or directions.shape[0] != 2 or directions.shape[-1] != 3
                or available.shape != directions.shape[:-1]
                or not np.isfinite(directions).all() or not np.isin(available, [0, 1]).all()
                or type(ports) is not int or ports < 1
                or type(neighbors) is not int or neighbors < 1):
            raise ValueError("Explicit paired directions, availability and local sizes required")
        if any(available[eye].sum() < neighbors for eye in range(2)):
            raise ValueError("Each eye needs enough observed rays for a local neighborhood")
        active = directions[available.astype(bool)]
        if not np.allclose(np.linalg.norm(active, axis=1), 1, atol=1e-5):
            raise ValueError("Observed directions must be unit vectors")
        rng = np.random.default_rng(seed)
        choices, centers = [], []
        for _ in range(ports):
            eye = int(rng.integers(2))
            candidates = np.flatnonzero(available[eye])
            center = int(rng.choice(candidates))
            # Stable tie ordering; RF assignment is a seeded hypothesis, not soma-side inference.
            order = np.argsort(-(directions[eye, candidates] @ directions[eye, center]),
                               kind="stable")[:neighbors]
            rays = eye * directions.shape[1] + candidates[order]
            choices.append((rays[:, None] * 3 + np.arange(3)).reshape(-1))
            centers.append([eye, center])
        self.register_buffer("indices", torch.tensor(np.array(choices), dtype=torch.long))
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.long))
        self.weight = nn.Parameter(torch.empty(ports, neighbors * 3))
        nn.init.normal_(self.weight, std=.05 / np.sqrt(neighbors * 3))

    def forward(self, flattened):
        return (flattened[:, self.indices] * self.weight[None]).sum(-1)


class FrozenRateCircuit(nn.Module):
    """Fixed count-normalized recurrent rate dynamics with sparse matrix multiplication.

    Weights are positive engineering defaults. No neurotransmitter/receptor signs or
    measured time constants are claimed. Dynamics match the default SparseCircuit.
    """

    def __init__(self, graph, dt, incoming_totals=None):
        super().__init__()
        validate_graph(graph)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite timestep required")
        self.size = len(graph["ids"])
        incoming = np.bincount(graph["post"], weights=graph["weights"], minlength=self.size)
        if incoming_totals is not None:
            totals = np.asarray(incoming_totals, dtype=float)
            if (totals.shape != (self.size,) or not np.isfinite(totals).all()
                    or np.any(totals < incoming)):
                raise ValueError("Full-source incoming totals must cover every retained synapse")
            incoming = totals
        weights = graph["weights"] / np.maximum(incoming[graph["post"]], 1)
        matrix = torch.sparse_coo_tensor(np.array([graph["post"], graph["pre"]]),
                                         torch.tensor(weights, dtype=torch.float32),
                                         (self.size, self.size), check_invariants=True).coalesce()
        self.register_buffer("matrix", matrix)
        self.register_buffer("input_mask", torch.tensor(graph["roles"] == "input"))
        self.register_buffer("state_mask", torch.ones(self.size))
        self.register_buffer("output_ports", torch.tensor(np.flatnonzero(
            graph["roles"] == "output")))
        tau = .01 + torch.nn.functional.softplus(torch.tensor(-2.6))
        self.register_buffer("alpha", 1 - torch.exp(-dt / tau))

    def forward(self, drive, state):
        if (state.ndim != 2 or state.shape[1] != self.size or drive.shape != state.shape
                or not torch.isfinite(drive).all() or torch.any(drive[:, ~self.input_mask])):
            raise ValueError("External drive may enter declared input ports only")
        message = torch.sparse.mm(self.matrix, state.T).T
        state = (1 - self.alpha) * state + self.alpha * torch.tanh(message + drive)
        state = state * self.state_mask
        return state[:, self.output_ports], state


class LocalRetinalController(nn.Module):
    def __init__(self, graph, visual_ids, organ_groups, directions, available, organ_size,
                 *, output_signs, neighbors=12, seed=730101, dt=.01,
                 output_activation="tanh", incoming_totals=None):
        super().__init__()
        local = LocalRetinalLinear(directions, available, len(visual_ids), neighbors, seed)
        width = int(np.asarray(directions).size)
        self.adapter = ModalityPortAdapter(graph, visual_ids, organ_groups, organ_size, width,
                                          visual_adapter=local)
        # Static pose fitting initially learns retinal transduction only. Keep organ
        # adapters separate and frozen at zero, rather than silently fitting label shortcuts.
        for adapter in self.adapter.organ_adapters:
            nn.init.zeros_(adapter.weight)
            adapter.requires_grad_(False)
        self.circuit = FrozenRateCircuit(graph, dt, incoming_totals=incoming_totals)
        self.register_buffer("directions", torch.tensor(directions, dtype=torch.float32))
        self.register_buffer("feature_mean", torch.zeros(width))
        self.register_buffer("feature_scale", torch.ones(width))
        self.register_buffer("readout_gain", torch.tensor(1.))
        if (len(self.circuit.output_ports) != 2 or len(output_signs) != 2
                or set(output_signs) != {-1, 1}):
            raise ValueError("This pilot requires two explicitly ordered differential outputs")
        self.register_buffer("output_signs", torch.tensor(output_signs, dtype=torch.float32))
        if output_activation not in ("tanh", "linear"):
            raise ValueError("Explicit tanh or linear output activation required")
        self.output_activation = output_activation

    def initial_state(self, batch=1):
        return self.readout_gain.new_zeros((batch, self.circuit.size))

    def prepare_drive(self, observation):
        if set(observation) != SensoryFeatureEncoder.fields:
            raise ValueError("Only the declared actor fields are accepted")
        rgb, valid = observation["ray_values"], observation["ray_valid"]
        if (rgb.shape[1:] != self.directions.shape or valid.shape != rgb.shape[:-1]
                or observation["ray_directions"].shape != rgb.shape
                or not torch.equal(observation["ray_directions"], self.directions.expand_as(rgb))
                or not torch.isfinite(rgb).all() or torch.any((rgb < 0) | (rgb > 1))
                or not torch.all((valid == 0) | (valid == 1))):
            raise ValueError("Eye directions, values or availability differ from calibration")
        mask = valid[..., None].expand_as(rgb).flatten(1)
        features = ((rgb.flatten(1) - self.feature_mean) / self.feature_scale) * mask
        return self.adapter(features, valid.flatten(1).any(1, keepdim=True).float(),
                            observation["organs"], observation["organ_valid"])

    def advance(self, drive, state):
        outputs, state = self.circuit(drive, state)
        # No visual residual, output bias or direct internal-neuron injection.
        steering = (outputs @ self.output_signs) * self.readout_gain
        if self.output_activation == "tanh":
            steering = torch.tanh(steering)
        return steering, state

    def forward(self, observation, state):
        return self.advance(self.prepare_drive(observation), state)
