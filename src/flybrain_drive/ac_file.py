"""Read-only latest-snapshot transport for AC installations lacking native Python sockets."""

import os
import time
from pathlib import Path

from .ac_tcp import PROTOCOL
from .ac_transport import MAX_DATAGRAM, decode_telemetry


class FileTelemetryClient:
    def __init__(self, directory, *, timeout=2., poll_seconds=.005, clock=None):
        if not 0 < poll_seconds <= timeout <= 60:
            raise ValueError("Positive bounded polling and receive timeout required")
        self.directory = Path(directory)
        self.timeout, self.poll_seconds = timeout, poll_seconds
        self.clock = clock or time.monotonic
        self.last_sequence = None
        self.last_packet = None
        self.transient_read_retries = 0
        self.stale_snapshot_retries = 0
        self.hello = self._read("producer.json")
        if (self.hello.raw.get("protocol") != PROTOCOL
                or self.hello.raw.get("kind") != "hello"
                or self.hello.raw.get("read_only") is not True
                or self.hello.raw.get("transport") != "local_latest_file"
                or not isinstance(self.hello.raw.get("producer_session_id"), str)
                or not self.hello.raw["producer_session_id"]):
            raise ValueError("Explicit read-only file producer handshake required")
        self.session_id = self.hello.raw["producer_session_id"]

    def _read(self, name):
        with (self.directory / name).open("rb") as stream:
            data = stream.read(MAX_DATAGRAM + 1)
        if len(data) > MAX_DATAGRAM:
            raise ValueError("File telemetry frame exceeds limit")
        return decode_telemetry(data, self.clock())

    def receive(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                packet = self._read("latest.json")
            except PermissionError as error:
                sharing = (getattr(error, 'winerror', None) in (32, 33)
                           or os.name == 'nt' and error.errno == 13)
                if not sharing or time.monotonic() >= deadline:
                    raise
                self.transient_read_retries += 1
                time.sleep(self.poll_seconds)
                continue
            if packet.raw.get("producer_session_id") != self.session_id:
                raise ValueError("Producer session changed; explicit recovery required")
            if self.last_sequence is not None and packet.sequence < self.last_sequence:
                raise ValueError("Telemetry sequence reset")
            if packet.sequence > 0 and packet.sequence != self.last_sequence:
                self.last_sequence = packet.sequence
                self.last_packet = packet
                return packet
            if time.monotonic() >= deadline:
                raise TimeoutError("No new file telemetry before receive deadline")
            time.sleep(self.poll_seconds)


    def snapshot(self, *, maximum_age=.05):
        """Latest diagnostic only; unchanged bytes retain their first receipt timestamp."""
        if (isinstance(maximum_age, bool) or not isinstance(maximum_age, (int, float))
                or not 0 < maximum_age <= .05):
            raise ValueError('Diagnostic snapshot age must remain within50ms')
        deadline = time.monotonic()+min(self.timeout, .01)
        while True:
            try:
                packet = self._read('latest.json')
            except PermissionError as error:
                sharing = (getattr(error, 'winerror', None) in (32, 33)
                           or os.name == 'nt' and error.errno == 13)
                if not sharing or time.monotonic() >= deadline:
                    raise
                self.transient_read_retries += 1
                time.sleep(min(self.poll_seconds, .002))
                continue
            if packet.raw.get('producer_session_id') != self.session_id:
                raise ValueError('Producer session changed; explicit recovery required')
            if packet.sequence <= 0 or (self.last_sequence is not None
                                       and packet.sequence < self.last_sequence):
                raise ValueError('Telemetry sequence reset or unavailable')
            if self.last_sequence == packet.sequence:
                if self.last_packet is None or packet.wire_bytes != self.last_packet.wire_bytes:
                    raise ValueError('Unchanged diagnostic sequence has different raw bytes')
                packet = self.last_packet
            else:
                self.last_sequence, self.last_packet = packet.sequence, packet
            if not 0 <= self.clock()-packet.received_monotonic_seconds <= maximum_age:
                if time.monotonic() < deadline:
                    self.stale_snapshot_retries += 1
                    time.sleep(min(self.poll_seconds, .002))
                    continue
                raise TimeoutError('Original diagnostic receipt is stale; never re-date a snapshot')
            return packet
