"""Read existing native AC physics mappings; no game plugin, network or actuator access."""

import math
import re
import struct
import time
from dataclasses import dataclass
from types import MappingProxyType

from .ac_image_buffer import ExistingNamedMapping

# Prefix through localAngularVel from pinned AC Gym sim_info.py, pack=4.
# Explicit little-endian 32-bit values avoid host wchar/long ABI differences.
PHYSICS_LAYOUT = (
    ("packetId", "i", 1), ("gas", "f", 1), ("brake", "f", 1), ("fuel", "f", 1),
    ("gear", "i", 1), ("rpms", "i", 1), ("steerAngle", "f", 1), ("speedKmh", "f", 1),
    ("velocity", "f", 3), ("accG", "f", 3), ("wheelSlip", "f", 4),
    ("wheelLoad", "f", 4), ("wheelsPressure", "f", 4), ("wheelAngularSpeed", "f", 4),
    ("tyreWear", "f", 4), ("tyreDirtyLevel", "f", 4), ("tyreCoreTemperature", "f", 4),
    ("camberRAD", "f", 4), ("suspensionTravel", "f", 4), ("drs", "f", 1),
    ("tc", "f", 1), ("heading", "f", 1), ("pitch", "f", 1), ("roll", "f", 1),
    ("cgHeight", "f", 1), ("carDamage", "f", 5), ("numberOfTyresOut", "i", 1),
    ("pitLimiterOn", "i", 1), ("abs", "f", 1), ("kersCharge", "f", 1),
    ("kersInput", "f", 1), ("autoShifterOn", "i", 1), ("rideHeight", "f", 2),
    ("turboBoost", "f", 1), ("ballast", "f", 1), ("airDensity", "f", 1),
    ("airTemp", "f", 1), ("roadTemp", "f", 1), ("localAngularVel", "f", 3),
)
PHYSICS_PREFIX = struct.Struct("<" + "".join(code * count for _, code, count in PHYSICS_LAYOUT))
BODY_FIELDS = frozenset({"gas", "brake", "steerAngle", "speedKmh", "accG", "wheelLoad",
    "wheelsPressure", "wheelAngularSpeed", "tyreCoreTemperature", "suspensionTravel",
    "localAngularVel"})


@dataclass(frozen=True)
class NativePhysicsSample:
    packet_id: int
    fields: object
    prefix_bytes: bytes
    received_monotonic_seconds: float
    source_time_seconds: None = None
    simulation_clock_verified: bool = False
    atomicity_verified: bool = False


def decode_physics_prefix(data, received_monotonic_seconds):
    if (not isinstance(data, bytes) or len(data) != PHYSICS_PREFIX.size
            or not math.isfinite(received_monotonic_seconds) or received_monotonic_seconds < 0):
        raise ValueError("Exact native physics prefix and finite receipt clock required")
    values = PHYSICS_PREFIX.unpack(data)
    fields, offset, packet_id = {}, 0, values[0]
    if packet_id < 0:
        raise ValueError("Native packet ID wrapped/reset or ABI differs")
    for name, _, count in PHYSICS_LAYOUT:
        selected = values[offset:offset + count]
        if not all(math.isfinite(v) for v in selected):
            raise ValueError("Nonfinite native physics field: " + name)
        if name in BODY_FIELDS:
            fields[name] = selected[0] if count == 1 else selected
        offset += count
    return NativePhysicsSample(packet_id, MappingProxyType(fields), data,
                               float(received_monotonic_seconds))


def repeated_snapshot(mapping, *, attempts=8, clock=None):
    """Matching repeated copies are a consistency check, not an engine-provided seqlock."""
    if type(attempts) is not int or not 1 <= attempts <= 64:
        raise ValueError("Bounded consistency attempts required")
    for _ in range(attempts):
        first, second = mapping.read(), mapping.read()
        if first == second:
            return decode_physics_prefix(second, (clock or time.monotonic)())
    raise RuntimeError("Native physics changed during all repeated-copy attempts")


def decode_static_identity_prefix(data):
    # Stop before playerName: never read/copy personal driver names.
    if not isinstance(data, bytes) or len(data) != 200:
        raise ValueError("Exact 200-byte nonpersonal static prefix required")
    result = {}
    for name, start, stop in (("shared_memory_version", 0, 30), ("ac_version", 30, 60),
                               ("car_model", 68, 134), ("track", 134, 200)):
        text = data[start:stop].decode("utf-16-le").split("\0", 1)[0]
        if not text or re.fullmatch(r"[a-zA-Z0-9_. -]{1,32}", text) is None:
            raise ValueError("Native static identity is empty or ABI differs: " + name)
        result[name] = text
    return result


class NativePhysicsReader:
    def __init__(self, *, timeout=2., poll_seconds=.005, clock=None):
        if not 0 < poll_seconds <= timeout <= 60:
            raise ValueError("Positive bounded native receive deadline required")
        self.timeout, self.poll_seconds, self.last_packet = timeout, poll_seconds, None
        self.clock = clock or time.monotonic
        self.mapping = ExistingNamedMapping("Local\\acpmf_physics", PHYSICS_PREFIX.size)

    def receive(self):
        deadline = time.monotonic() + self.timeout
        while True:
            sample = repeated_snapshot(self.mapping, clock=self.clock)
            if self.last_packet is not None and sample.packet_id < self.last_packet:
                raise ValueError("Native physics sequence reset; explicit new session required")
            if sample.packet_id != self.last_packet:
                self.last_packet = sample.packet_id
                return sample
            if time.monotonic() >= deadline:
                raise TimeoutError("Native physics has no new packet before deadline")
            time.sleep(self.poll_seconds)

    def close(self):
        self.mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
