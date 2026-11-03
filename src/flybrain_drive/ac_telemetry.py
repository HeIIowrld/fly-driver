"""Explicit calibration of selected AC fields; raw pose/progress never become actor inputs."""

from dataclasses import dataclass

import numpy as np

from .sensor_contract import BODY_SCHEMA, BodySample, SampleTime, readonly_array

ALLOWED_KEYS = {
    "angular_velocity": ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
    "specific_force": ("accelX", "accelY"),
    "steering_angle": ("steerAngle",),
    "wheel_angular_velocity": tuple("wheel_speed_" + s for s in ("fl", "fr", "rl", "rr")),
    "suspension_position": tuple(s + "_damper_linear_potentiometer"
                                 for s in ("fl", "fr", "rl", "rr")),
    "tyre_core_temperature": tuple(s + "_tire_temperature_core"
                                    for s in ("fl", "fr", "rl", "rr")),
    "wheel_normal_load": tuple(s + "_wheel_load" for s in ("fl", "fr", "rl", "rr")),
}


@dataclass(frozen=True)
class ChannelCalibration:
    channel: str
    keys: tuple
    matrix: object
    evidence_id: str
    offset: object = None

    def __post_init__(self):
        matrix = readonly_array(self.matrix)
        rows = matrix.shape[0] if matrix.ndim == 2 else 0
        offset = readonly_array(np.zeros(rows) if self.offset is None else self.offset)
        keys = tuple(self.keys)
        if (self.channel not in ALLOWED_KEYS or not isinstance(self.evidence_id, str)
                or not self.evidence_id or not keys
                or len(set(keys)) != len(keys) or not set(keys) <= set(ALLOWED_KEYS[self.channel])
                or matrix.ndim != 2 or matrix.shape[1] != len(keys)
                or matrix.shape[0] not in BODY_SCHEMA[self.channel][2]
                or offset.shape != (matrix.shape[0],)):
            raise ValueError("Permitted channel, transform and calibration evidence required")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    def convert(self, packet, sample_time):
        if (not isinstance(sample_time, SampleTime)
                or sample_time.received_monotonic_seconds != packet.received_monotonic_seconds):
            raise ValueError("Conversion must retain the actual packet receipt timestamp")
        values, available = [], []
        for key in self.keys:
            raw = packet.raw.get(key)
            present = type(raw) in (int, float) and np.isfinite(raw)
            values.append(float(raw) if present else 0.)
            available.append(present)
        dependencies = self.matrix != 0
        valid = dependencies.any(1) & np.all(~dependencies | np.array(available)[None], axis=1)
        converted = self.matrix @ values + self.offset
        if self.channel in ("tyre_core_temperature", "wheel_normal_load"):
            valid &= converted >= 0
        unit, frame, _ = BODY_SCHEMA[self.channel]
        # Until native units/signs are measured, this remains a declared conversion model.
        return BodySample(self.channel, np.where(valid, converted, 0), valid, sample_time,
                          unit, frame, "declared_sensor_model")


class LapTimeClock:
    """Candidate clock reconstruction; scale and live-game validation remain explicit.

    A normal lap transition preserves elapsed time. Reset, skipped laps and backwards
    readings are rejected rather than silently clearing the neural state.
    """

    def __init__(self, seconds_per_unit):
        if not np.isfinite(seconds_per_unit) or seconds_per_unit <= 0:
            raise ValueError("An explicit positive clock scale is required")
        self.scale = seconds_per_unit
        self.previous = None
        self.origin = 0.

    def update(self, packet):
        lap, current, last = (packet.raw.get(k) for k in
                              ("completedLaps", "currentTime", "iLastTime"))
        if (type(lap) is not int or lap < 0 or type(current) not in (float, int)
                or not np.isfinite(current) or current < 0):
            raise ValueError("Lap clock fields are unavailable")
        origin = self.origin
        if self.previous is not None:
            old_sequence, old_lap, old_time = self.previous
            if packet.sequence <= old_sequence:
                raise ValueError("Clock packet reordered or reset")
            if lap == old_lap:
                if current < old_time:
                    raise ValueError("Clock moved backwards; explicit episode handling required")
            elif lap == old_lap + 1:
                if type(last) not in (float, int) or not np.isfinite(last) or last < old_time:
                    raise ValueError("Lap boundary clock cannot be reconstructed")
                origin += last * self.scale
            else:
                raise ValueError("Reset or skipped laps require explicit clock/session recovery")
        self.previous = (packet.sequence, lap, current)
        self.origin = origin
        return origin + current * self.scale
