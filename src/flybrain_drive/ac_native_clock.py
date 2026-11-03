"""Read-only native AC graphics timing diagnostics; never actor route inputs.

The pinned upstream layout is Windows UTF-16 with four-byte packing. This prefix
ends at sessionTimeLeft, before distanceTraveled and world coordinates. Lap counts
and times are diagnostics only; no simulation clock or live-control permission is
inferred from these fields.
"""

import base64
import hashlib
import math
import struct
import time
from dataclasses import asdict, dataclass

from .ac_image_buffer import ExistingNamedMapping

GRAPHICS_PREFIX = struct.Struct('<iii120siiiiif')
STATUS_NAMES = {0: 'off', 1: 'replay', 2: 'live', 3: 'pause'}


@dataclass(frozen=True)
class NativeGraphicsTiming:
    packet_id: int
    status: str
    session: int
    completed_laps: int
    current_time_raw: int
    last_time_raw: int
    best_time_raw: int
    session_time_left_raw: float
    received_monotonic_seconds: float
    prefix_bytes: bytes

    def metadata(self):
        result = asdict(self)
        raw = result.pop('prefix_bytes')
        result.update(prefix_base64=base64.b64encode(raw).decode(),
                      prefix_sha256=hashlib.sha256(raw).hexdigest())
        return result


def decode_graphics_timing(data, received_monotonic_seconds):
    if (not isinstance(data, bytes) or len(data) != GRAPHICS_PREFIX.size
            or not math.isfinite(received_monotonic_seconds)
            or received_monotonic_seconds < 0):
        raise ValueError('Exact graphics timing prefix and host receipt clock required')
    packet, status, session, _, laps, _, current, last, best, left = GRAPHICS_PREFIX.unpack(data)
    if (packet < 0 or status not in STATUS_NAMES or session not in range(-1, 7)
            or laps < 0 or not math.isfinite(left)):
        raise ValueError('Graphics timing ABI or session state differs')
    return NativeGraphicsTiming(packet, STATUS_NAMES[status], session, laps, current, last,
                                best, left, float(received_monotonic_seconds), data)


class NativeGraphicsTimingReader:
    """A snapshot may be repeated/stale; caller must retain and inspect packet IDs."""

    def __init__(self):
        self.mapping = ExistingNamedMapping('Local\\acpmf_graphics', GRAPHICS_PREFIX.size)

    def snapshot(self):
        for _ in range(8):
            first, second = self.mapping.read(), self.mapping.read()
            if first == second:
                return decode_graphics_timing(second, time.monotonic())
        raise RuntimeError('Graphics timing changed during repeated-copy attempts')

    def close(self):
        self.mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
