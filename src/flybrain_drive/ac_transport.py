"""Modern reader for the pinned AC Gym UDP telemetry protocol; no actuator sender."""

import ipaddress
import json
import math
import numbers
import socket
import time
from dataclasses import dataclass
from types import MappingProxyType

MAX_DATAGRAM = 65507


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate telemetry JSON key")
        result[key] = value
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite telemetry number")
    if type(value) is int and abs(value) > 2**53 - 1:
        raise ValueError("Telemetry integer exceeds exact interoperable range")
    return value


@dataclass(frozen=True)
class TelemetryPacket:
    sequence: int
    source_monotonic_seconds: float
    received_monotonic_seconds: float
    raw: object
    wire_bytes: bytes


def decode_telemetry(data, received_monotonic_seconds):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_DATAGRAM:
        raise ValueError("Expected one bounded UDP JSON datagram")
    def invalid_constant(_):
        raise ValueError("Nonfinite telemetry JSON constant")

    try:
        raw = json.loads(data.decode("utf8"), object_pairs_hook=_object,
                         parse_constant=invalid_constant)
        frozen = _freeze(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid telemetry JSON") from error
    if (not isinstance(raw, dict) or type(raw.get("steps")) is not int or raw["steps"] < 0
            or type(raw.get("timestamp_ac")) not in (int, float)
            or not math.isfinite(raw["timestamp_ac"]) or raw["timestamp_ac"] < 0
            or not math.isfinite(received_monotonic_seconds) or received_monotonic_seconds < 0):
        raise ValueError("Telemetry needs an update sequence and separate finite clocks")
    return TelemetryPacket(raw["steps"], float(raw["timestamp_ac"]),
                           float(received_monotonic_seconds), frozen, data)


class TelemetryClient:
    """Explicit subscription takes the upstream server's single client slot.

    Source and receipt clocks remain separate. No control or reset messages are sent.
    Closing only closes our socket: it cannot disconnect a replacement client.
    """

    def __init__(self, host, port=2345, timeout=1.):
        ipaddress.IPv4Address(host)
        if type(port) is not int or not 1 <= port <= 65535 or not 0 < timeout <= 60:
            raise ValueError("Explicit peer port and bounded timeout required")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.connect((host, port))
        self.socket.settimeout(timeout)
        self.last_sequence = None
        self.subscribed = False

    def subscribe(self):
        if self.subscribed:
            raise ValueError("Already subscribed; reconnect is an explicit new session")
        self.socket.send(b"connect")
        reply = self.socket.recv(MAX_DATAGRAM)
        if reply != b"identified":
            raise ValueError("AC Gym handshake was not identified")
        self.subscribed = True

    def receive(self):
        if not self.subscribed:
            raise ValueError("An explicit subscription is required")
        data = self.socket.recv(MAX_DATAGRAM)
        received = time.monotonic()
        if data == b"disconnect":
            raise ConnectionError("Producer disconnected; explicit session handling required")
        packet = decode_telemetry(data, received)
        if self.last_sequence is not None and packet.sequence <= self.last_sequence:
            raise ValueError("Duplicate/reordered telemetry or producer counter reset")
        self.last_sequence = packet.sequence
        return packet

    def close(self):
        self.socket.close()
        self.subscribed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def encode_control_request(steering, throttle, brake, sequence, *, steering_sign):
    """Dry codec only. This is a requested command, never an applied-control acknowledgement."""
    if (not all(isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)
                for v in (steering, throttle, brake))
            or not -1 <= steering <= 1 or not 0 <= throttle <= 1 or not 0 <= brake <= 1
            or type(steering_sign) is not int or steering_sign not in (-1, 1)
            or type(sequence) is not int or sequence < 0):
        raise ValueError("Explicit normalized controls, steering sign and source sequence required")
    return json.dumps({"steer": steering * steering_sign, "acc": 2 * throttle - 1,
        "brake": 2 * brake - 1, "server_steps": sequence, "enable_gear_shift": False,
        "shift_up": False, "shift_down": False}, allow_nan=False).encode("utf8")
