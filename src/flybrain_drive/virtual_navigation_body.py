"""Versioned vehicle speed/wheel feedback; no claim of measured fly sensory tuning."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ac_native_physics import decode_physics_prefix
from .graph import sha256
from .sensor_contract import readonly_array, readonly_mask
from .virtual_body import (
    FEATURE_NAMES,
    NativeVirtualBodySample,
    VirtualBodyEncoder,
    decode_virtual_body_prefix,
)

NAVIGATION_FEATURE_NAMES = (
    *FEATURE_NAMES,
    "vehicle.speed_magnitude_m_s",
    *(f"wheel_rotation.native_{i}_rad_s" for i in range(4)),
)


@dataclass(frozen=True)
class NativeNavigationBodySample:
    legacy: NativeVirtualBodySample
    speed_magnitude_m_s: float
    wheel_rotation_native_rad_s: tuple


@dataclass(frozen=True)
class NavigationBodyObservation:
    episode_id: str
    packet_id: int
    original_received_monotonic_seconds: float
    decision_monotonic_seconds: float
    values: np.ndarray
    valid: np.ndarray
    contract_sha256: str
    schema: str = "flybrain.virtual_navigation_body.observation.v2"
    feature_names: tuple = NAVIGATION_FEATURE_NAMES
    physiological_registration_verified: bool = False


def decode_navigation_body_prefix(data, received_monotonic_seconds):
    sample = decode_physics_prefix(data, received_monotonic_seconds)
    if sample.fields["speedKmh"] < 0:
        raise ValueError("Native unsigned speed magnitude cannot be negative")
    return NativeNavigationBodySample(
        decode_virtual_body_prefix(data, received_monotonic_seconds),
        sample.fields["speedKmh"] / 3.6,
        tuple(sample.fields["wheelAngularSpeed"]),
    )


class NavigationBodyEncoder:
    def __init__(self, contract_file, *, native_identity):
        path = Path(contract_file)
        contract = json.loads(path.read_text(encoding="utf8"))
        if (
            contract["schema"] != "flybrain.virtual_navigation_body.contract.v2"
            or contract["legacy_feature_count"] != 16
            or contract["feature_count"] != 21
            or sha256(contract["legacy_contract"]) != contract["legacy_contract_sha256"]
        ):
            raise ValueError("Explicit unchanged legacy16 plus vehicle feedback5 contract required")
        self.legacy = VirtualBodyEncoder(
            contract["legacy_contract"], native_identity=native_identity
        )
        self.contract_sha256 = sha256(path)
        self.last, self.episode = None, None

    def encode(self, sample, *, episode_id, decision_monotonic_seconds):
        if not isinstance(sample, NativeNavigationBodySample):
            raise ValueError("Versioned native navigation sample required")
        extra = np.array([sample.speed_magnitude_m_s, *sample.wheel_rotation_native_rad_s])
        if extra.shape != (5,) or not np.isfinite(extra).all() or extra[0] < 0:
            raise ValueError("Finite unsigned speed and four native wheel rates required")
        if (
            self.episode == episode_id
            and self.last is not None
            and self.last.legacy.packet_id == sample.legacy.packet_id
        ):
            if self.last.legacy.prefix_sha256 != sample.legacy.prefix_sha256:
                raise ValueError("Repeated native packet has different bytes")
            sample = self.last  # Preserve the first receipt and actual original five values.
            extra = np.array([sample.speed_magnitude_m_s, *sample.wheel_rotation_native_rad_s])
        old = self.legacy.encode(
            sample.legacy,
            episode_id=episode_id,
            decision_monotonic_seconds=decision_monotonic_seconds,
        )
        fresh = (
            decision_monotonic_seconds - old.original_received_monotonic_seconds
            <= self.legacy.maximum_age
        )
        values = np.concatenate([old.values, extra if fresh else np.zeros(5)])
        valid = np.concatenate([old.valid, np.full(5, fresh)])
        self.last, self.episode = sample, episode_id
        return NavigationBodyObservation(
            episode_id,
            old.packet_id,
            old.original_received_monotonic_seconds,
            old.decision_monotonic_seconds,
            readonly_array(values),
            readonly_mask(valid),
            self.contract_sha256,
        )
