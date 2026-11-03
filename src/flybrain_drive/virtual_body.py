"""Versioned vehicle-body feedback; calibrated tilt is distinct from angular rate."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, decode_physics_prefix
from .graph import sha256
from .sensor_contract import readonly_array, readonly_mask

FEATURE_NAMES = (
    "attitude.pitch_nose_up_rad",
    "attitude.roll_left_up_rad",
    "angular_body.roll_rad_s",
    "angular_body.pitch_nose_down_rad_s",
    "angular_body.yaw_left_rad_s",
    "linear_acceleration_body.forward_m_s2",
    "linear_acceleration_body.left_m_s2",
    "linear_acceleration_body.up_m_s2",
    *(f"wheel_load_fraction.native_{index}" for index in range(4)),
    *(f"tyre_core_temperature.native_{index}" for index in range(4)),
)


@dataclass(frozen=True)
class NativeVirtualBodySample:
    packet_id: int
    received_monotonic_seconds: float
    prefix_sha256: str
    pitch_native: float
    roll_native: float
    angular_native: tuple
    acceleration_native_g: tuple
    wheel_load_native: tuple
    tyre_temperature_native: tuple


@dataclass(frozen=True)
class VirtualBodyObservation:
    episode_id: str
    packet_id: int
    original_received_monotonic_seconds: float
    decision_monotonic_seconds: float
    values: np.ndarray
    valid: np.ndarray
    contract_sha256: str
    schema: str = "flybrain.virtual_body.observation.v1"
    feature_names: tuple = FEATURE_NAMES
    simulation_clock_verified: bool = False
    physiological_registration_verified: bool = False


def decode_virtual_body_prefix(data, received_monotonic_seconds):
    """New explicit decoder; legacy BODY_FIELDS and checkpoints keep their contract."""
    legacy = decode_physics_prefix(data, received_monotonic_seconds)
    fields, offset = {}, 0
    unpacked = PHYSICS_PREFIX.unpack(data)
    for name, _, count in PHYSICS_LAYOUT:
        if name in ("pitch", "roll"):
            fields[name] = unpacked[offset]
        offset += count
    return NativeVirtualBodySample(
        legacy.packet_id,
        legacy.received_monotonic_seconds,
        hashlib.sha256(data).hexdigest(),
        fields["pitch"],
        fields["roll"],
        legacy.fields["localAngularVel"],
        legacy.fields["accG"],
        legacy.fields["wheelLoad"],
        legacy.fields["tyreCoreTemperature"],
    )


class VirtualBodyEncoder:
    """No differentiation or invented samples; reuse retains the first receipt age."""

    def __init__(self, contract_file, *, native_identity):
        path = Path(contract_file)
        contract = json.loads(path.read_text(encoding="utf8"))
        if any(
            native_identity.get(key) != value for key, value in contract["native_identity"].items()
        ):
            raise ValueError("Native car/version identity differs from reviewed body contract")
        review_path = Path(contract["geometry_review"])
        if sha256(review_path) != contract["geometry_review_sha256"]:
            raise ValueError("Virtual-body geometry evidence changed")
        review = json.loads(review_path.read_text(encoding="utf8"))
        if review["status"] != "completed" or len(review["recordings"]) < 2:
            raise ValueError("Two retained geometry comparisons required")
        if contract["schema"] != "flybrain.virtual_body.contract.v1":
            raise ValueError("Unsupported virtual-body contract")
        tilt = [
            all(row["tilt"][name]["passed"] for row in review["recordings"])
            for name in ("pitch_nose_up", "roll_left_up")
        ]
        angular = [
            all(row["angular"][name]["passed"] for row in review["recordings"])
            for name in ("roll", "pitch", "yaw")
        ]
        if contract["tilt_valid"] != tilt or contract["angular_valid"] != angular:
            raise ValueError("Declared body masks differ from geometry evidence")
        if contract["acceleration_body_registered"] is not False:
            raise ValueError("Linear acceleration axis and gravity registration remain unverified")
        if contract["angular_native_indices"] != [2, 0, 1] or contract["tilt_signs"] != [1, -1]:
            raise ValueError("Body mapping differs from reviewed convention")
        age = contract["maximum_host_age_seconds"]
        if not np.isfinite(age) or not 0 < age <= 0.05:
            raise ValueError("Explicit bounded host receipt age required")
        self.contract_sha256 = sha256(path)
        self.tilt_valid, self.angular_valid = tilt, angular
        self.maximum_age = age
        self.reset()

    def reset(self):
        self.episode = None
        self.last_sample = None
        self.last_decision = None

    def encode(self, sample, *, episode_id, decision_monotonic_seconds):
        if not isinstance(sample, NativeVirtualBodySample) or not isinstance(episode_id, str):
            raise ValueError("Explicit native virtual-body sample and episode identity required")
        if not episode_id or not np.isfinite(decision_monotonic_seconds):
            raise ValueError("Nonempty episode and finite host decision clock required")
        if episode_id != self.episode:
            self.reset()
            self.episode = episode_id
        if self.last_decision is not None and decision_monotonic_seconds < self.last_decision:
            raise ValueError("Virtual-body decision clock reversed")
        if self.last_sample is not None:
            if sample.packet_id < self.last_sample.packet_id:
                raise ValueError("Native packet reset without explicit episode boundary")
            if sample.packet_id == self.last_sample.packet_id:
                if sample.prefix_sha256 != self.last_sample.prefix_sha256:
                    raise ValueError("Repeated native packet has different bytes")
                sample = self.last_sample
            elif sample.received_monotonic_seconds < self.last_sample.received_monotonic_seconds:
                raise ValueError("Native receipt clock reversed")
        age = decision_monotonic_seconds - sample.received_monotonic_seconds
        if age < 0:
            raise ValueError("Native sample is from after decision")
        values = np.zeros(len(FEATURE_NAMES), dtype=float)
        valid = np.zeros(len(FEATURE_NAMES), dtype=bool)
        values[:2] = [sample.pitch_native, -sample.roll_native]
        valid[:2] = self.tilt_valid
        values[2:5] = np.asarray(sample.angular_native)[[2, 0, 1]]
        valid[2:5] = self.angular_valid
        # Never call accG linear acceleration without its frame/gravity registration.
        load = np.array(sample.wheel_load_native)
        if (load >= 0).all() and load.sum() > 0:
            values[8:12] = load / load.sum()
            valid[8:12] = True
        values[12:16] = sample.tyre_temperature_native
        valid[12:16] = True
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite virtual-body feature")
        if age > self.maximum_age:
            valid[:] = False
        values[~valid] = 0.0
        self.last_sample, self.last_decision = sample, float(decision_monotonic_seconds)
        return VirtualBodyObservation(
            episode_id,
            sample.packet_id,
            sample.received_monotonic_seconds,
            float(decision_monotonic_seconds),
            readonly_array(values),
            readonly_mask(valid),
            self.contract_sha256,
        )
