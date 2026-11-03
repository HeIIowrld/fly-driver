"""View-local transduction with fusion confined to shared recurrent structural connectivity."""

import copy

import numpy as np
import torch
from torch import nn

from .modality_ports import ModalityPortAdapter
from .multiview_sensory import MultiViewPacker
from .vehicle_circuit import SignedRateCircuit, VehicleReadout


class MultiViewLocalLinear(nn.Module):
    """Every visual port reads exactly one declared view and one local neighborhood.

    Per-ray inputs are RGB, temporal RGB contrast and their validity flags. Contrast
    is not optical-flow velocity. Age only attenuates the port's own source. Input
    assignment is an engineering hypothesis, not anatomical receptive-field mapping.
    Unit coordinates define local neighborhoods. A native pixel frontend may supply
    a unit index embedding; that does not establish calibrated physical body rays.
    """

    def __init__(self, directions, available, port_sources, *, neighbors, seed, age_scale):
        super().__init__()
        directions, available, source = map(np.asarray, (directions, available, port_sources))
        if (directions.ndim != 3 or directions.shape[-1] != 3 or directions.shape[0] < 2
                or available.shape != directions.shape[:-1]
                or not np.isfinite(directions).all() or not np.isin(available, [0, 1]).all()
                or source.ndim != 1 or not len(source) or source.dtype.kind not in "iu"
                or set(source.tolist()) != set(range(len(directions)))
                or type(neighbors) is not int or neighbors < 1
                or not np.isfinite(age_scale) or age_scale <= 0
                or np.any(available.sum(1) < neighbors)
                or not np.allclose(np.linalg.norm(directions[available.astype(bool)], axis=1),
                                   1, atol=1e-5)):
            raise ValueError("Views, valid rays, source coverage and local calibration required")
        self.shape = directions.shape[:2]
        rng = np.random.default_rng(seed)
        indices, centers = [], []
        for view in source:
            candidates = np.flatnonzero(available[view])
            center = int(rng.choice(candidates))
            order = np.argsort(-(directions[view, candidates] @ directions[view, center]),
                               kind="stable")[:neighbors]
            rays = int(view) * self.shape[1] + candidates[order]
            indices.append((rays[:, None] * 8 + np.arange(8)).ravel())
            centers.append([int(view), center])
        self.register_buffer("indices", torch.tensor(np.array(indices), dtype=torch.long))
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.long))
        self.register_buffer("port_sources", torch.tensor(source, dtype=torch.long))
        self.weight = nn.Parameter(torch.empty(len(source), neighbors * 8))
        nn.init.normal_(self.weight, std=.05 / np.sqrt(neighbors * 8))
        self.age_scale = age_scale
        self.width = int(np.prod(self.shape) * 8 + self.shape[0])

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError("View-local feature layout differs from calibration")
        ages = features[:, -self.shape[0]:][:, self.port_sources]
        return ((features[:, self.indices] * self.weight[None]).sum(-1)
                * torch.exp(-ages / self.age_scale))


class MultiViewCircuitController(nn.Module):
    """All cross-view interactions take place after local input-port injection."""

    def __init__(self, graph, visual_ids, organ_groups, directions, available, port_sources,
                 signs, incoming_totals, output_groups, steering_signs, sensory_calibration,
                 *, neighbors=12, seed=800101, dt=.01, tau=.05,
                 photometric_normalization=False, sparse_backend="coo",
                 pedal_mode='brake_attenuated', organ_validity_mode='learned_drive',
                 readout_gain_parameterization='softplus',
                 activation_gradient='exact', surrogate_negative_slope=.05,
                 visual_age_scale=None):
        super().__init__()
        calibration = copy.deepcopy(sensory_calibration)
        count = np.asarray(directions).shape[0]
        if (len(calibration["view_order"]) != count
                or len(set(calibration["view_order"])) != count
                or set(calibration["view_hashes"]) != set(calibration["view_order"])
                or any(not isinstance(h, str) or len(h) != 64
                       or any(c not in "0123456789abcdef" for c in h)
                       for h in calibration["view_hashes"].values())
                or len(port_sources) != len(visual_ids)):
            raise ValueError("Ordered source calibrations and visual port allocation required")
        self.sensory_calibration = calibration
        if visual_age_scale is not None and (
                isinstance(visual_age_scale, bool) or not np.isfinite(visual_age_scale)
                or not calibration['max_view_age'] <= visual_age_scale <= 1.):
            raise ValueError('Visual confidence scale must lie between hard freshness and 1s')
        if type(photometric_normalization) is not bool:
            raise ValueError("Explicit photometric normalization mode required")
        self.photometric_normalization = photometric_normalization
        if photometric_normalization:
            shape = (*np.asarray(directions).shape[:2], 6)
            self.register_buffer("photometric_mean", torch.zeros(shape))
            self.register_buffer("photometric_scale", torch.ones(shape))
            self.register_buffer("photometric_seen", torch.zeros((*shape[:2], 2)))
        local = MultiViewLocalLinear(directions, available, port_sources, neighbors=neighbors,
                                     seed=seed, age_scale=calibration['max_view_age']
                                     if visual_age_scale is None else visual_age_scale)
        self.adapter = ModalityPortAdapter(graph, visual_ids, organ_groups,
            calibration["organ_size"], local.width, visual_adapter=local,
            organ_validity_mode=organ_validity_mode)
        self.circuit = SignedRateCircuit(graph, signs, incoming_totals, dt=dt, tau=tau,
                                         sparse_backend=sparse_backend,
                                         activation_gradient=activation_gradient,
                                         surrogate_negative_slope=surrogate_negative_slope)
        self.readout = VehicleReadout(graph, output_groups, steering_signs, pedal_mode=pedal_mode,
                                     gain_parameterization=readout_gain_parameterization)

    def initial_state(self, batch=1):
        return self.circuit.alpha.new_zeros((batch, self.circuit.size))

    def prepare_drive(self, observation):
        if set(observation) != MultiViewPacker.fields:
            raise ValueError("Only source-specific actor fields are accepted")
        rgb, contrast, valid, contrast_valid, ages = (observation[k] for k in
            ("view_rgb", "view_contrast", "view_valid", "contrast_valid", "view_age"))
        shape = self.adapter.visual_adapter.shape
        if (rgb.ndim != 4 or rgb.shape[1:] != (*shape, 3) or contrast.shape != rgb.shape
                or valid.shape != rgb.shape[:-1] or contrast_valid.shape != valid.shape
                or ages.shape != (len(rgb), shape[0])
                or not all(torch.isfinite(v).all()
                           for v in (rgb, contrast, valid, contrast_valid, ages))
                or torch.any((rgb < 0) | (rgb > 1)) or torch.any(contrast.abs() > 1)
                or not all(((v == 0) | (v == 1)).all() for v in (valid, contrast_valid))
                or torch.any(contrast_valid > valid) or torch.any(ages < 0)):
            raise ValueError("Bounded independent view values, masks and ages required")
        stale = ages > self.sensory_calibration["max_view_age"]
        if torch.any(valid * stale[..., None]):
            raise ValueError("Stale visual sources must be invalidated before inference")
        if self.photometric_normalization:
            if (not torch.isfinite(self.photometric_mean).all()
                    or not torch.isfinite(self.photometric_scale).all()
                    or torch.any(self.photometric_scale <= 0)
                    or not torch.isin(self.photometric_seen,
                                      self.photometric_seen.new_tensor([0, 1])).all()):
                raise ValueError("Finite frozen photometric calibration required")
            values = torch.cat([rgb, contrast], dim=-1)
            masks = torch.stack([valid, contrast_valid], -1) * self.photometric_seen
            masks = masks.repeat_interleave(3, dim=-1)
            values = ((values - self.photometric_mean) / self.photometric_scale).clamp(-5, 5)
            values = values * masks
            # Validity gates evidence; it is not a trainable constant brightness signal.
            features = torch.cat([values, torch.zeros_like(masks[..., :2])], -1).flatten(1)
        else:
            features = torch.cat([rgb * valid[..., None], contrast * contrast_valid[..., None],
                                  valid[..., None], contrast_valid[..., None]], dim=-1).flatten(1)
        features = torch.cat([features, ages], dim=1)
        return self.adapter(features, valid.flatten(1).any(1, keepdim=True).float(),
                            observation["organs"], observation["organ_valid"])

    def advance(self, drive, state):
        outputs, state = self.circuit(drive, state)
        return self.readout(outputs), state

    def validate_training_chunk(self, drive, state):
        if (state.ndim != 2 or drive.ndim != 3 or drive.shape[1:] != state.shape
                or state.shape[1] != self.circuit.size
                or not torch.isfinite(drive).all() or not torch.isfinite(state).all()
                or torch.any((state < 0) | (state > 1))
                or torch.any(drive[..., ~self.circuit.input_mask])
                or self.circuit.matrix.requires_grad
                or not torch.isfinite(self.circuit.matrix.values()).all()
                or not torch.isfinite(self.circuit.alpha).all()
                or torch.any((self.circuit.alpha <= 0) | (self.circuit.alpha > 1))
                or not torch.isfinite(self.circuit.state_mask).all()
                or torch.any((self.circuit.state_mask < 0) | (self.circuit.state_mask > 1))):
            raise ValueError('Finite fixed-graph training chunk and input-port-only drive required')

    def advance_training_chunk(self, drive, state):
        outputs, state = self.circuit._advance_tensor(drive, state)
        return self.readout._decode_tensor(outputs), state

    def validate_training_result(self, prediction, state):
        if (not torch.isfinite(state).all() or torch.any((state < 0) | (state > 1))
                or not torch.isfinite(prediction).all()
                or torch.any(prediction[..., 0].abs() > 1)
                or torch.any((prediction[..., 1:] < 0) | (prediction[..., 1:] > 1))):
            raise ValueError('Training chunk produced invalid state or controls')

    def forward(self, observation, state):
        return self.advance(self.prepare_drive(observation), state)


class MultiViewCircuitRunner:
    def __init__(self, model, packer):
        if model.sensory_calibration != packer.calibration:
            raise ValueError("Multi-view model and packer calibrations differ")
        self.model, self.packer = model, packer
        self.model.eval()
        self.reset()

    def reset(self):
        self.episode, self.state = None, None
        self.packer.reset()

    def step(self, observation):
        packed = self.packer.pack(observation)
        if observation.episode_id != self.episode:
            self.state = self.model.initial_state()
            self.episode = observation.episode_id
        tensors = {key: torch.tensor(value, device=self.model.circuit.alpha.device)[None]
                   for key, value in packed.items()}
        with torch.no_grad():
            requests, self.state = self.model(tensors, self.state)
        return requests[0].cpu().numpy(), packed
