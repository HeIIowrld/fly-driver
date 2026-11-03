"""Functional organ-inspired encoding, not measured FeCO/haltere physiology."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .sensor_contract import ActorObservation, readonly_array


@dataclass(frozen=True)
class PositionOrganSpec:
    sensor: str
    channels: int
    minimum: float
    maximum: float
    tuning_width: float
    velocity_scale: float
    populations: int = 7

    def __post_init__(self):
        if self.sensor not in ("steering_angle", "suspension_position"):
            raise ValueError("Position organs require steering or suspension position")
        allowed = (1, 2) if self.sensor == "steering_angle" else (4,)
        if (self.channels not in allowed or type(self.populations) is not int
                or self.populations < 2):
            raise ValueError("Invalid organ channel/population count")
        if not np.isfinite([
            self.minimum, self.maximum, self.tuning_width, self.velocity_scale
        ]).all():
            raise ValueError("Organ calibration must be finite")
        if (self.minimum >= self.maximum or self.tuning_width <= 0 or self.velocity_scale <= 0):
            raise ValueError("Organ calibration range/width/velocity scale invalid")


@dataclass(frozen=True)
class OrganEncoding:
    values: np.ndarray
    valid: np.ndarray
    names: tuple[str, ...]
    fidelity: str = "functional_vehicle_to_organ_surrogate_not_measured_neurons"


@dataclass(frozen=True)
class ScalarOrganSpec:
    """Vehicle temperature/load population code; no claim of homologous fly organs.

    Ranges describe the encoding, not optimal tyre temperatures or load limits.
    Physical units and FL/FR/RL/RR ordering come from BodySample's contract.
    """

    sensor: str
    minimum: float
    maximum: float
    tuning_width: float
    change_scale: float
    populations: int = 7

    def __post_init__(self):
        if self.sensor not in ("tyre_core_temperature", "wheel_normal_load"):
            raise ValueError("Scalar organs require declared tyre temperature or wheel load")
        if (type(self.populations) is not int or self.populations < 2
                or not np.isfinite([self.minimum, self.maximum,
                                    self.tuning_width, self.change_scale]).all()
                or self.minimum < 0 or self.minimum >= self.maximum
                or min(self.tuning_width, self.change_scale) <= 0):
            raise ValueError("Invalid scalar organ calibration")

    @property
    def channels(self):
        return 4

    @property
    def velocity_scale(self):
        return self.change_scale


class BodyOrganEncoder:
    """State persists between decisions; only new samples update movement channels."""

    def __init__(self, positions, rotation_scale=None, max_sample_age=0.15, *, scalars=()):
        self.positions = tuple(positions)
        self.scalars = tuple(scalars)
        if (not all(isinstance(p, PositionOrganSpec) for p in self.positions)
                or not all(isinstance(p, ScalarOrganSpec) for p in self.scalars)):
            raise ValueError("Explicit position/scalar organ specifications required")
        if len({p.sensor for p in self.positions}) != len(self.positions):
            raise ValueError("Duplicate position organ specifications")
        if len({p.sensor for p in self.scalars}) != len(self.scalars):
            raise ValueError("Duplicate scalar organ specifications")
        if not np.isfinite(max_sample_age) or max_sample_age <= 0:
            raise ValueError("Maximum sample age must be finite and positive")
        if rotation_scale is not None and (not np.isfinite(rotation_scale) or rotation_scale <= 0):
            raise ValueError("Rotation scale must be finite and positive")
        self.rotation_scale, self.max_sample_age = rotation_scale, max_sample_age
        self.reset()

    @property
    def feature_count(self):
        return (sum(p.channels * (p.populations + 2) for p in self.positions + self.scalars)
                + (6 if self.rotation_scale is not None else 0))

    @property
    def feature_names(self):
        names = []
        for spec in self.positions + self.scalars:
            kind = "level" if isinstance(spec, ScalarOrganSpec) else "position"
            labels = [f"{kind}_{i}" for i in range(spec.populations)] + ["increase", "decrease"]
            names.extend(f"{spec.sensor}.{c}.{label}" for c in range(spec.channels)
                         for label in labels)
        if self.rotation_scale is not None:
            names.extend(f"rotation.{axis}.{sign}" for axis in "xyz" for sign in ("+", "-"))
        return tuple(names)

    @property
    def calibration_sha256(self):
        # Legacy checkpoints retain their original contract. New scalar contracts
        # seal tuning as well as width, including position/rotation calibration.
        if not self.scalars:
            return None
        settings = {"version": "vehicle-scalar-organs-v1",
                    "positions": [asdict(p) for p in self.positions],
                    "scalars": [asdict(p) for p in self.scalars],
                    "rotation_scale": self.rotation_scale,
                    "max_sample_age": self.max_sample_age}
        return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf8")).hexdigest()

    def reset(self):
        self.episode = None
        self.last_sequence = -1
        self.last_time = -1.0
        self.last_host_time = -1.0
        self.history = {}

    def encode(self, observation):
        if not isinstance(observation, ActorObservation):
            raise ValueError("Organ encoder accepts ActorObservation only")
        if observation.episode_id != self.episode:
            self.reset()
            self.episode = observation.episode_id
        if (observation.sequence_id <= self.last_sequence
                or observation.simulation_seconds < self.last_time
                or observation.decision_monotonic_seconds < self.last_host_time):
            raise ValueError("Observation sequence/time moved backwards; reset episode explicitly")
        # Validate all groups before changing the numerical state.
        body = {b.name: b for b in observation.body}
        for spec in self.positions + self.scalars:
            sample = body.get(spec.sensor)
            if sample is not None and len(sample.values) != spec.channels:
                raise ValueError("Sensor dimensions differ from declared organ calibration")
            previous = self.history.get(spec.sensor)
            if sample is not None and previous is not None:
                stamp = sample.time.simulation_seconds
                if stamp < previous[0]:
                    raise ValueError("Organ sample time moved backwards")
                if stamp == previous[0] and (
                    not np.array_equal(sample.values, previous[1])
                    or not np.array_equal(sample.valid, previous[2])
                ):
                    raise ValueError("Reused sample timestamp has changed values/validity")
        values, valid, names = [], [], []
        for spec in self.positions + self.scalars:
            sample = body.get(spec.sensor)
            width = spec.populations + 2
            features = np.zeros((spec.channels, width))
            mask = np.zeros_like(features, dtype=bool)
            if sample is not None and observation.ages()[spec.sensor] <= self.max_sample_age:
                centers = np.linspace(spec.minimum, spec.maximum, spec.populations)
                features[:, :spec.populations] = np.exp(
                    -0.5 * ((sample.values[:, None] - centers) / spec.tuning_width) ** 2
                )
                mask[:, :spec.populations] = sample.valid[:, None]
                previous = self.history.get(spec.sensor)
                stamp = sample.time.simulation_seconds
                if previous is not None and stamp == previous[0]:
                    features[:, -2:], mask[:, -2:] = previous[3], previous[4]
                elif previous is not None:
                    velocity = (sample.values - previous[1]) / (stamp - previous[0])
                    features[:, -2] = np.tanh(np.maximum(velocity, 0) / spec.velocity_scale)
                    features[:, -1] = np.tanh(np.maximum(-velocity, 0) / spec.velocity_scale)
                    mask[:, -2:] = (sample.valid & previous[2])[:, None]
                self.history[spec.sensor] = (
                    stamp, sample.values.copy(), sample.valid.copy(),
                    features[:, -2:].copy(), mask[:, -2:].copy(),
                )
            else:
                # A gap invalidates movement history; return is not a movement measurement.
                self.history.pop(spec.sensor, None)
            values.extend((features * mask).ravel())
            valid.extend(mask.ravel())
            kind = "level" if isinstance(spec, ScalarOrganSpec) else "position"
            labels = [f"{kind}_{i}" for i in range(spec.populations)] + ["increase", "decrease"]
            names.extend(f"{spec.sensor}.{c}.{label}" for c in range(spec.channels)
                         for label in labels)
        if self.rotation_scale is not None:
            sample = body.get("angular_velocity")
            rotation, mask = np.zeros((3, 2)), np.zeros((3, 2), dtype=bool)
            if sample is not None and observation.ages()["angular_velocity"] <= self.max_sample_age:
                rotation[:, 0] = np.tanh(np.maximum(sample.values, 0) / self.rotation_scale)
                rotation[:, 1] = np.tanh(np.maximum(-sample.values, 0) / self.rotation_scale)
                mask[:] = sample.valid[:, None]
            values.extend((rotation * mask).ravel())
            valid.extend(mask.ravel())
            names.extend(f"rotation.{axis}.{sign}" for axis in "xyz" for sign in ("+", "-"))
        self.last_sequence = observation.sequence_id
        self.last_time = observation.simulation_seconds
        self.last_host_time = observation.decision_monotonic_seconds
        return OrganEncoding(readonly_array(values), readonly_array(valid, bool), tuple(names))
