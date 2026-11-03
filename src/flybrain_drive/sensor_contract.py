"""Backend-neutral sensor boundary. SI units and clocks are explicit, not inferred."""

import re
from dataclasses import dataclass

import numpy as np

# Coordinates must be converted by the backend before constructing these samples.
# Body axes: X forward, Y left, Z up; wheel order: FL, FR, RL, RR.
BODY_SCHEMA = {
    "angular_velocity": ("rad/s", "body", (3,)),
    "specific_force": ("m/s2", "body", (3,)),
    "steering_angle": ("rad", "roadwheel", (1, 2)),
    "wheel_angular_velocity": ("rad/s", "wheel_axles", (4,)),
    "suspension_position": ("m", "compression_positive", (4,)),
    "suspension_velocity": ("m/s", "compression_positive", (4,)),
    "tyre_core_temperature": ("K", "tyre_core", (4,)),
    "wheel_normal_load": ("N", "contact_normal_compression_positive", (4,)),
}


def readonly_array(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Sensor arrays must be finite, including invalid placeholders")
    result.setflags(write=False)
    return result


def readonly_mask(value):
    original = np.asarray(value)
    if (original.dtype.kind not in "biuf" or not np.isfinite(original).all()
            or not np.isin(original, [0, 1]).all()):
        raise ValueError("Validity must contain finite boolean/zero-one values")
    return readonly_array(original, bool)


@dataclass(frozen=True)
class SampleTime:
    simulation_seconds: float
    received_monotonic_seconds: float

    def __post_init__(self):
        if not np.isfinite([self.simulation_seconds, self.received_monotonic_seconds]).all():
            raise ValueError("Sample clocks must be finite")
        if min(self.simulation_seconds, self.received_monotonic_seconds) < 0:
            raise ValueError("Sample clocks must be nonnegative")


@dataclass(frozen=True)
class BodySample:
    name: str
    values: np.ndarray
    valid: np.ndarray
    time: SampleTime
    unit: str
    frame: str
    provenance: str

    def __post_init__(self):
        if self.name not in BODY_SCHEMA:
            raise ValueError("Unknown/privileged body channel: " + self.name)
        unit, frame, widths = BODY_SCHEMA[self.name]
        if (self.unit, self.frame) != (unit, frame):
            raise ValueError("Backend must explicitly convert sensor units and coordinates")
        if self.provenance not in ("instrumented", "declared_sensor_model"):
            raise ValueError("Body input requires instrumented or declared sensor-model provenance")
        values = readonly_array(self.values)
        if not isinstance(self.time, SampleTime):
            raise ValueError("Sensor timestamp must be SampleTime")
        valid = readonly_mask(self.valid)
        if values.ndim != 1 or len(values) not in widths or valid.shape != values.shape:
            raise ValueError("Body value/validity shape does not match the sensor schema")
        if self.name in ("tyre_core_temperature", "wheel_normal_load") and np.any(
                values[valid] < 0):
            raise ValueError("Valid absolute temperature and compressive load must be nonnegative")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class EyeSample:
    side: str
    frame_id: int
    response: np.ndarray
    valid: np.ndarray
    time: SampleTime
    eye_map_sha256: str

    def __post_init__(self):
        if self.side not in ("left", "right") or type(self.frame_id) is not int:
            raise ValueError("Eye requires left/right and an integer exposure ID")
        if self.frame_id < 0:
            raise ValueError("Exposure ID must be nonnegative")
        if (len(self.eye_map_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.eye_map_sha256)):
            raise ValueError("Eye-map provenance must be a lowercase SHA-256")
        if not isinstance(self.time, SampleTime):
            raise ValueError("Eye timestamp must be SampleTime")
        values, valid = readonly_array(self.response), readonly_mask(self.valid)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("Eye response must be rays x channels")
        if valid.shape != (len(values),):
            raise ValueError("Eye validity must cover each ray")
        object.__setattr__(self, "response", values)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class VisualViewSample:
    """Source-specific retinal RGB with a declared exposure, never a capture-count alias."""

    source_id: str
    frame_id: int
    response: np.ndarray
    valid: np.ndarray
    time: SampleTime
    calibration_sha256: str
    provenance: str

    def __post_init__(self):
        if (not isinstance(self.source_id, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", self.source_id) is None
                or type(self.frame_id) is not int or self.frame_id < 0):
            raise ValueError("View requires a source ID and nonnegative integer exposure ID")
        if (not isinstance(self.time, SampleTime)
                or not isinstance(self.calibration_sha256, str)
                or len(self.calibration_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.calibration_sha256)
                or self.provenance not in ("instrumented", "declared_sensor_model")):
            raise ValueError("View requires declared timing, calibration SHA256 and provenance")
        response, valid = readonly_array(self.response), readonly_mask(self.valid)
        if (response.ndim != 2 or response.shape[1] != 3 or not len(response)
                or valid.shape != (len(response),) or np.any((response < 0) | (response > 1))):
            raise ValueError("View response must be bounded rays x RGB with per-ray validity")
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class AppliedControl:
    steering: float
    throttle: float
    brake: float
    time: SampleTime

    def __post_init__(self):
        if not isinstance(self.time, SampleTime):
            raise ValueError("Applied control timestamp must be SampleTime")
        if not np.isfinite([self.steering, self.throttle, self.brake]).all():
            raise ValueError("Applied controls must be finite")
        if not (-1 <= self.steering <= 1 and 0 <= self.throttle <= 1 and 0 <= self.brake <= 1):
            raise ValueError("Applied steering/throttle/brake outside normalized ranges")


@dataclass(frozen=True)
class ActorObservation:
    episode_id: str
    sequence_id: int
    simulation_seconds: float
    decision_monotonic_seconds: float
    eyes: tuple[EyeSample, ...]
    body: tuple[BodySample, ...]
    previous_applied: AppliedControl | None = None
    views: tuple[VisualViewSample, ...] = ()

    def __post_init__(self):
        SampleTime(self.simulation_seconds, self.decision_monotonic_seconds)
        if not self.episode_id or type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("Observation needs an episode and nonnegative integer sequence")
        eyes, body, views = tuple(self.eyes), tuple(self.body), tuple(self.views)
        if any(not isinstance(e, EyeSample) for e in eyes):
            raise ValueError("Actor eyes must be EyeSample objects")
        if any(not isinstance(b, BodySample) for b in body):
            raise ValueError("Actor body must be BodySample objects")
        if (any(not isinstance(v, VisualViewSample) for v in views) or eyes and views):
            raise ValueError("Use declared views or legacy eyes, never mixed visual schemas")
        if len({v.source_id for v in views}) != len(views):
            raise ValueError("Duplicate view sources")
        if len({e.side for e in eyes}) != len(eyes) or len({b.name for b in body}) != len(body):
            raise ValueError("Duplicate eye/body groups")
        if (self.previous_applied is not None
                and not isinstance(self.previous_applied, AppliedControl)):
            raise ValueError("Previous command must be a declared applied control")
        samples = [*eyes, *body, *views]
        if self.previous_applied is not None:
            samples.append(self.previous_applied)
        for sample in samples:
            if (sample.time.simulation_seconds > self.simulation_seconds
                    or sample.time.received_monotonic_seconds > self.decision_monotonic_seconds):
                raise ValueError("Actor cannot use future or not-yet-received samples")
        object.__setattr__(self, "eyes", eyes)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "views", views)

    def ages(self):
        """Sample ages in simulation time; never subtract across clock domains."""
        return {
            **{"eye_" + e.side: self.simulation_seconds - e.time.simulation_seconds
               for e in self.eyes},
            **{b.name: self.simulation_seconds - b.time.simulation_seconds for b in self.body},
            **{"view_" + v.source_id: self.simulation_seconds - v.time.simulation_seconds
               for v in self.views},
        }


@dataclass(frozen=True)
class TrainingOnly:
    """Separate container; never accepted as an ActorObservation body/eye sample."""

    episode_id: str
    sequence_id: int
    teacher_action: tuple[float, ...] | None = None
    route_progress: float | None = None
    lateral_error: float | None = None
