"""Read-only TCP transport for our AC telemetry app, separate from upstream Gym UDP."""

import ipaddress
import socket
import time

from .ac_transport import MAX_DATAGRAM, decode_telemetry

PROTOCOL = "flybrain.ac.telemetry.v1"


class TcpTelemetryClient:
    def __init__(self, host, port=2348, timeout=1., *, clock=time.monotonic):
        ipaddress.IPv4Address(host)
        if type(port) is not int or not 1 <= port <= 65535 or not 0 < timeout <= 60:
            raise ValueError("Explicit peer and bounded timeout required")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clock = clock
        self.socket.settimeout(timeout)
        try:
            self.socket.connect((host, port))
        except OSError:
            self.socket.close()
            raise
        self.buffer = bytearray()
        self.last_sequence = None
        self.session_id = None
        self.hello = None

    def _line(self):
        while True:
            position = self.buffer.find(b"\n")
            if position >= 0:
                if position > MAX_DATAGRAM:
                    raise ValueError("TCP telemetry frame exceeds limit")
                data = bytes(self.buffer[:position])
                del self.buffer[:position + 1]
                return data
            if len(self.buffer) > MAX_DATAGRAM:
                raise ValueError("TCP telemetry frame exceeds limit")
            chunk = self.socket.recv(8192)
            if not chunk:
                raise ConnectionError(
                    "Producer closed TCP stream; explicit session recovery required")
            self.buffer.extend(chunk)

    def subscribe(self):
        if self.hello is not None:
            raise ValueError("Already subscribed")
        # Reuse bounded strict JSON decoding, including duplicate/finite validation.
        hello = decode_telemetry(self._line(), self.clock())
        if (hello.raw.get("protocol") != PROTOCOL or hello.raw.get("kind") != "hello"
                or hello.raw.get("read_only") is not True
                or not isinstance(hello.raw.get("producer_session_id"), str)
                or not hello.raw["producer_session_id"]):
            raise ValueError("Unsupported telemetry producer handshake")
        self.hello = hello
        self.session_id = hello.raw["producer_session_id"]

    def receive(self):
        if self.hello is None:
            raise ValueError("Explicit subscription required")
        packet = decode_telemetry(self._line(), self.clock())
        if packet.raw.get("producer_session_id") != self.session_id:
            raise ValueError("Producer session changed inside stream")
        if self.last_sequence is not None and packet.sequence <= self.last_sequence:
            raise ValueError("Telemetry sequence reordered/reset")
        self.last_sequence = packet.sequence
        return packet

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def record_packet(packet):
    """Exact wire bytes and receipt clock; no clock-domain subtraction or action acknowledgement."""
    import base64
    import hashlib

    return {"sequence": packet.sequence,
            "received_monotonic_seconds": packet.received_monotonic_seconds,
            "wire_base64": base64.b64encode(packet.wire_bytes).decode(),
            "wire_sha256": hashlib.sha256(packet.wire_bytes).hexdigest()}
