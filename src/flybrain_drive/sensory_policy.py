"""Validated eye/organ observations to a recurrent policy; no route-truth input."""

import numpy as np
import torch
from torch import nn

from .sensor_contract import ActorObservation


class ObservationPacker:
    def __init__(self, eyes, organs, *, specific_force_scale, wheel_speed_scale,
                 max_sample_age=.15):
        self.eyes = {eye.side: eye for eye in eyes}
        if set(self.eyes) != {"left", "right"} or len(eyes) != 2:
            raise ValueError("Packer requires exactly two declared eye maps")
        if not np.isfinite([specific_force_scale, wheel_speed_scale, max_sample_age]).all():
            raise ValueError("Sensor scales/age must be finite")
        if min(specific_force_scale, wheel_speed_scale, max_sample_age) <= 0:
            raise ValueError("Sensor scales/age must be positive")
        self.organs = organs
        self.scales = (specific_force_scale, wheel_speed_scale)
        self.max_sample_age = max_sample_age
        self.episode = None
        self.exposures = {}

    @property
    def organ_size(self):
        return self.organs.feature_count

    def pack(self, observation):
        if not isinstance(observation, ActorObservation):
            raise ValueError("Packer requires ActorObservation")
        if observation.views:
            raise ValueError("Source-specific views require the multi-view packer")
        new_episode = observation.episode_id != self.episode
        history = {} if new_episode else self.exposures
        for eye in observation.eyes:
            geometry = self.eyes[eye.side]
            if (eye.eye_map_sha256 != geometry.sha256
                    or eye.response.shape != (len(geometry.directions_body), 3)
                    or np.any((eye.response < 0) | (eye.response > 1))):
                raise ValueError("Eye response/map differs from the declared RGB geometry")
            previous = history.get(eye.side)
            if previous is not None:
                same = eye.frame_id == previous.frame_id
                if eye.frame_id < previous.frame_id:
                    raise ValueError("Exposure ID moved backwards")
                if same and (eye.time.simulation_seconds != previous.time.simulation_seconds
                             or not np.array_equal(eye.response, previous.response)
                             or not np.array_equal(eye.valid, previous.valid)):
                    raise ValueError("Reused exposure changed its timestamp/values/validity")
                if not same and eye.time.simulation_seconds <= previous.time.simulation_seconds:
                    raise ValueError("New exposure requires increasing simulation timestamp")
        encoded = self.organs.encode(observation)
        n = max(len(eye.directions_body) for eye in self.eyes.values())
        rays, directions, mask = np.zeros((2, n, 3)), np.zeros((2, n, 3)), np.zeros((2, n))
        ages = np.full(2, self.max_sample_age * 2)
        supplied = {e.side: e for e in observation.eyes}
        for i, side in enumerate(("left", "right")):
            geometry = self.eyes[side]
            directions[i, :len(geometry.directions_body)] = geometry.directions_body
            eye = supplied.get(side)
            if eye is not None:
                ages[i] = observation.ages()["eye_" + side]
                if ages[i] <= self.max_sample_age:
                    rays[i, :len(eye.response)] = eye.response * eye.valid[:, None]
                    mask[i, :len(eye.response)] = eye.valid
        auxiliary, auxiliary_valid = np.zeros(7), np.zeros(7)
        body = {b.name: b for b in observation.body}
        for name, scale, region in (
            ("specific_force", self.scales[0], slice(0, 3)),
            ("wheel_angular_velocity", self.scales[1], slice(3, 7)),
        ):
            sample = body.get(name)
            if sample is not None and observation.ages()[name] <= self.max_sample_age:
                auxiliary[region] = np.clip(sample.values / scale, -100, 100) * sample.valid
                auxiliary_valid[region] = sample.valid
        control, control_valid = np.zeros(3), np.zeros(1)
        applied = observation.previous_applied
        if (applied is not None
                and observation.simulation_seconds - applied.time.simulation_seconds
                <= self.max_sample_age):
            control[:] = [applied.steering, applied.throttle, applied.brake]
            control_valid[:] = 1
        self.episode = observation.episode_id
        self.exposures = {e.side: e for e in observation.eyes}
        return {name: np.asarray(value, dtype=np.float32) for name, value in {
            "ray_values": rays, "ray_directions": directions, "ray_valid": mask,
            "eye_age": np.minimum(ages, 10 * self.max_sample_age),
            "organs": encoded.values, "organ_valid": encoded.valid,
            "auxiliary": auxiliary, "auxiliary_valid": auxiliary_valid,
            "applied_control": control, "applied_valid": control_valid,
        }.items()}


class SensoryFeatureEncoder(nn.Module):
    """General learned baseline encoder; not a biological visual circuit."""

    fields = {"ray_values", "ray_directions", "ray_valid", "eye_age", "organs", "organ_valid",
              "auxiliary", "auxiliary_valid", "applied_control", "applied_valid"}

    def __init__(self, organ_size, eye_map_hashes, grid_shape=None, organ_calibration_sha256=None):
        super().__init__()
        if type(organ_size) is not int or organ_size < 1:
            raise ValueError("Expected a positive declared organ feature count")
        self.organ_size = organ_size
        if organ_calibration_sha256 is not None and (
            not isinstance(organ_calibration_sha256, str)
            or len(organ_calibration_sha256) != 64
            or any(c not in "0123456789abcdef" for c in organ_calibration_sha256)
        ):
            raise ValueError("Organ calibration must be a SHA256 digest")
        self.organ_calibration_sha256 = organ_calibration_sha256
        if (set(eye_map_hashes) != {"left", "right"}
                or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                       for h in eye_map_hashes.values())):
            raise ValueError("Policy requires explicit eye-map hashes for both eyes")
        self.eye_map_hashes = dict(eye_map_hashes)
        self.grid_shape = None if grid_shape is None else tuple(grid_shape)
        if self.grid_shape is not None and (
            len(self.grid_shape) != 2 or any(type(v) is not int or v < 2 for v in self.grid_shape)
        ):
            raise ValueError("Retinotopic grid requires declared integer rows/columns >=2")
        self.ray = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.grid = None if grid_shape is None else nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.Tanh(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.Tanh(),
            nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(150 + 2 * organ_size + (96 if self.grid is not None else 0), 64), nn.Tanh()
        )

    def forward(self, observation):
        if set(observation) != self.fields:
            raise ValueError("Sensory policy accepts only the declared actor tensor fields")
        rgb, directions, valid = (observation[k].float() for k in
                                  ("ray_values", "ray_directions", "ray_valid"))
        if (rgb.ndim != 4 or rgb.shape[1] != 2 or rgb.shape[-1] != 3
                or directions.shape != rgb.shape or valid.shape != rgb.shape[:-1]):
            raise ValueError("Ray tensors must be batch x two eyes x rays x RGB/direction")
        batch = len(rgb)
        for key, width in (("organs", self.organ_size), ("organ_valid", self.organ_size),
                           ("auxiliary", 7), ("auxiliary_valid", 7), ("eye_age", 2),
                           ("applied_control", 3), ("applied_valid", 1)):
            if observation[key].shape != (batch, width):
                raise ValueError("Declared actor feature width mismatch: " + key)
        side = torch.tensor([-1., 1.], device=rgb.device).view(1, 2, 1, 1)
        side = side.expand(batch, 2, rgb.shape[2], 1)
        embedding = self.ray(torch.cat([rgb * valid[..., None], directions, side], dim=-1))
        mean = (embedding * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]
        maximum = embedding.masked_fill(~valid.bool()[..., None], -2).max(2).values
        maximum = torch.where(valid.bool().any(2)[..., None], maximum, 0)
        organs, organ_mask = observation["organs"].float(), observation["organ_valid"].float()
        aux, aux_mask = observation["auxiliary"].float(), observation["auxiliary_valid"].float()
        control_mask = observation["applied_valid"].float()
        parts = [
            mean.flatten(1), maximum.flatten(1), valid.mean(2), observation["eye_age"].float(),
            organs * organ_mask, organ_mask, aux * aux_mask, aux_mask,
            observation["applied_control"].float() * control_mask, control_mask,
        ]
        if self.grid is not None:
            height, width = self.grid_shape
            if rgb.shape[2] != height * width:
                raise ValueError("Ray count differs from the sealed retinotopic grid")
            image = (rgb * valid[..., None]).reshape(batch, 2, height, width, 3)
            image = image.permute(0, 1, 4, 2, 3).flatten(1, 2)
            spatial_mask = valid.reshape(batch, 2, height, width)
            parts.append(self.grid(torch.cat([image, spatial_mask], dim=1)))
        return self.fusion(torch.cat(parts, dim=1))


class SensoryPolicyRunner:
    """Single-agent persistent inference, with explicit episode state reset."""

    def __init__(self, policy, packer):
        if (not isinstance(policy.encoder, SensoryFeatureEncoder)
                or policy.encoder.organ_size != packer.organ_size
                or policy.encoder.organ_calibration_sha256 != packer.organs.calibration_sha256
                or policy.encoder.eye_map_hashes != {
                    side: geometry.sha256 for side, geometry in packer.eyes.items()
                }):
            raise ValueError("Policy encoder and sensory contract do not match")
        grid = policy.encoder.grid_shape
        if grid is not None and any(len(g.directions_body) != grid[0]*grid[1]
                                    for g in packer.eyes.values()):
            raise ValueError("Each eye must match the sealed retinotopic grid")
        self.policy, self.packer = policy, packer
        self.policy.eval()
        self.episode, self.state = None, None

    def step(self, observation):
        packed = self.packer.pack(observation)
        if self.episode != observation.episode_id:
            self.state = self.policy.initial_state()
            self.episode = observation.episode_id
        device = self.policy.log_std.device
        tensors = {key: torch.tensor(value, device=device).unsqueeze(0)
                   for key, value in packed.items()}
        with torch.no_grad():
            mean, _, self.state = self.policy(tensors, self.state)
        return torch.tanh(mean)[0].cpu().numpy(), packed
