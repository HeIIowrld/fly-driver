"""Bounded, first-receipt-preserving native angular history; diagnostic only.

The stored vector stays in native axis order. This module does not validate an
axis, claim physical acquisition time, or feed an actor.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .ac_native_physics import NativePhysicsSample, decode_physics_prefix
from .causal_native_angular_average import integrate_received_angular_velocity


@dataclass(frozen=True)
class NativeAngularWindow:
    start: float
    end: float
    decision: float
    native_mean: tuple
    first_packet: int
    last_packet: int
    first_receipt: float
    last_receipt: float
    first_prefix_sha256: str
    last_prefix_sha256: str
    physical_time_verified: bool = False


class NativeAngularHistory:
    """Keep monotone distinct packets and retain original receipt on duplicates."""

    def __init__(self, *, capacity=512, max_gap=.05):
        if type(capacity) is not int or capacity < 2 or capacity > 65536:
            raise ValueError('Bounded packet capacity required')
        if isinstance(max_gap, bool) or not isinstance(max_gap, (int, float)) or not math.isfinite(max_gap) or not 0 < max_gap <= .05:
            raise ValueError('Bounded maximum packet gap required')
        self.capacity = capacity
        self.max_gap = float(max_gap)
        self.reset()

    def reset(self):
        self._rows = []
        self._last_observed_time = None

    def offer(self, sample):
        if not isinstance(sample, NativePhysicsSample):
            raise ValueError('Original decoded native packet required')
        checked = decode_physics_prefix(sample.prefix_bytes, sample.received_monotonic_seconds)
        if checked.packet_id != sample.packet_id or tuple(checked.fields['localAngularVel']) != tuple(sample.fields['localAngularVel']):
            raise ValueError('Native sample fields differ from raw bytes')
        digest = hashlib.sha256(sample.prefix_bytes).hexdigest()
        received = float(sample.received_monotonic_seconds)
        if self._last_observed_time is not None and received < self._last_observed_time:
            raise ValueError('Receipt moved backwards; explicit reset required')
        if self._rows:
            previous = self._rows[-1]
            if sample.packet_id < previous[0] or received < previous[1]:
                raise ValueError('Packet or receipt moved backwards; explicit reset required')
            if sample.packet_id == previous[0]:
                if digest != previous[3]:
                    raise ValueError('Repeated packet bytes changed')
                self._last_observed_time = received
                return False
            if received == previous[1]:
                raise ValueError('Distinct packets require advancing receipts')
        self._rows.append((sample.packet_id, received, tuple(checked.fields['localAngularVel']), digest))
        self._last_observed_time = received
        if len(self._rows) > self.capacity:
            self._rows.pop(0)
        return True

    def window(self, *, start, end, decision):
        if len(self._rows) < 2:
            raise ValueError('Insufficient distinct native packets')
        times = np.asarray([row[1] for row in self._rows], dtype=float)
        vectors = np.asarray([row[2] for row in self._rows], dtype=float)
        mean = integrate_received_angular_velocity(times, vectors, start=start, end=end,
                                                    decision=decision, max_gap=self.max_gap)
        available = np.searchsorted(times, decision, side='right')
        first = np.searchsorted(times[:available], start, side='right') - 1
        last = np.searchsorted(times[:available], end, side='left')
        a, b = self._rows[first], self._rows[last]
        return NativeAngularWindow(float(start), float(end), float(decision), tuple(mean.tolist()),
                                   a[0], b[0], a[1], b[1], a[3], b[3])
