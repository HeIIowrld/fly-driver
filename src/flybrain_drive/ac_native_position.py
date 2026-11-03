"""Native graphics position for privileged diagnostic bounds, never actor inputs.

Pinned AC SPageFileGraphic uses Windows UTF-16 and pack4: carCoordinates at252,
three floats. This is the original AC layout, not ACC's multicars array. The
original source is parsed/offset-checked in tests and stationary SDK comparison
is required before live use. Repeated copies are not an engine seqlock.
"""

import base64
import hashlib
import math
import struct
import time
from dataclasses import dataclass

from .ac_image_buffer import ExistingNamedMapping
from .ac_native_clock import GRAPHICS_PREFIX, decode_graphics_timing

POSITION_PREFIX = struct.Struct('<i248x3f')


@dataclass(frozen=True)
class NativeDiagnosticPosition:
    packet_id: int
    status: str
    coordinates_native_xyz_m: tuple
    received_perf_counter_seconds: float
    prefix_bytes: bytes

    def metadata(self):
        return dict(packet_id=self.packet_id, status=self.status,
            coordinates_native_xyz_m=list(self.coordinates_native_xyz_m),
            received_perf_counter_seconds=self.received_perf_counter_seconds,
            prefix_base64=base64.b64encode(self.prefix_bytes).decode(),
            prefix_sha256=hashlib.sha256(self.prefix_bytes).hexdigest(),
            actor_input=False, simulation_clock_verified=False, atomicity_verified=False)


def decode_diagnostic_position(data, received):
    if not isinstance(data, bytes) or len(data) != POSITION_PREFIX.size:
        raise ValueError('Exact264-byte original AC graphics prefix required')
    timing = decode_graphics_timing(data[:GRAPHICS_PREFIX.size], received)
    packet, *coordinates = POSITION_PREFIX.unpack(data)
    if not all(math.isfinite(v) and abs(v) < 1e7 for v in coordinates):
        raise ValueError('Finite diagnostic native coordinates required')
    return NativeDiagnosticPosition(packet, timing.status, tuple(coordinates), received, data)


class NativeDiagnosticPositionReader:
    def __init__(self, *, clock=time.perf_counter):
        self.mapping = ExistingNamedMapping('Local\\acpmf_graphics', POSITION_PREFIX.size)
        self.clock, self.last = clock, None

    def snapshot(self, *, maximum_age=.08):
        if not 0 < maximum_age <= .08:
            raise ValueError('Explicit native diagnostic age at most80ms required')
        for _ in range(8):
            first, second = self.mapping.read(), self.mapping.read()
            if first != second:
                continue
            now = self.clock()
            candidate = decode_diagnostic_position(second, now)
            if self.last is not None:
                if candidate.packet_id < self.last.packet_id:
                    raise ValueError('Native graphics sequence reset')
                if candidate.packet_id == self.last.packet_id:
                    if candidate.prefix_bytes != self.last.prefix_bytes:
                        raise ValueError('Unchanged graphics packet has different bytes')
                    candidate = self.last
            self.last = candidate
            if not 0 <= now-candidate.received_perf_counter_seconds <= maximum_age:
                raise TimeoutError('Native diagnostic graphics stale; original receipt retained')
            return candidate
        raise RuntimeError('Native graphics changed during repeated-copy attempts')

    def close(self):
        self.mapping.close()
