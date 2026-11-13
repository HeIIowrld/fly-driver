"""Read-only continuously drained TCP source with original receipt times retained."""
import socket
import threading
import time

from .ac_tcp import TcpTelemetryClient


class LatestTcpTelemetryClient:
    def __init__(self, *, host='127.0.0.1', port=2348, timeout=.15,
                 maximum_age=.05, clock=time.perf_counter, maximum_packets=50000,
                 source_admission=None):
        if not 0 < maximum_age <= .05 or not 0 < timeout <= 1 or not 10 <= maximum_packets <= 100000:
            raise ValueError('Bounded source receipt/timeout/archive budget required')
        self.client = TcpTelemetryClient(host, port, timeout=timeout, clock=clock)
        self.client.subscribe()
        self.session_id = self.client.session_id
        self.hello = self.client.hello
        self.clock, self.timeout, self.maximum_age = clock, timeout, maximum_age
        self.source_admission = source_admission
        self.maximum_packets = maximum_packets
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.error = None
        self.failure_observation = None
        self.cadence_failure_observation = None
        self.latest = None
        self.last_selected = None
        self.last_selected_source = None
        self.cadence_constrained_selections = 0
        self.packets = []
        self.thread = threading.Thread(target=self._receive, name='teacher-tcp-source', daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stopped.is_set():
                packet = self.client.receive()
                with self.condition:
                    if self.packets and packet.source_monotonic_seconds <= self.packets[-1].source_monotonic_seconds:
                        raise ValueError('TCP source time regressed/repeated')
                    if len(self.packets) >= self.maximum_packets:
                        raise RuntimeError('Bounded TCP raw archive budget exhausted')
                    self.packets.append(packet)
                    self.latest = packet
                    self.condition.notify_all()
        except BaseException as error:
            if not self.stopped.is_set():
                with self.condition:
                    stamp = self.clock()
                    latest = self.latest
                    self.failure_observation = dict(
                        kind=("socket_timeout" if isinstance(error, socket.timeout)
                              else "peer_closed_or_reset" if isinstance(error, ConnectionError)
                              else "source_error"),
                        error_type=type(error).__name__, error=str(error), observed_monotonic_seconds=stamp,
                        last_original_sequence=latest.sequence if latest is not None else None,
                        last_original_receipt=latest.received_monotonic_seconds if latest is not None else None,
                        pending_frame_bytes=len(self.client.buffer))
                    self.error = error
                    self.condition.notify_all()

    def receive(self, *, minimum_source_interval=0., maximum_source_interval=None):
        if (not 0 <= minimum_source_interval < 1 or maximum_source_interval is not None and
                not minimum_source_interval < maximum_source_interval <= 1):
            raise ValueError('Explicit bounded source cadence interval required')
        deadline = self.clock()+self.timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError('TCP source collector failed') from self.error
                packet = self.latest
                now = self.clock()
                if packet is not None and self.last_selected_source is not None and maximum_source_interval is not None:
                    in_window = [p for p in self.packets[-32:] if
                        p.sequence > self.last_selected and
                        minimum_source_interval <= p.source_monotonic_seconds-self.last_selected_source <= maximum_source_interval]
                    fresh = [p for p in in_window if 0 <= now-p.received_monotonic_seconds <= self.maximum_age]
                    eligible = [p for p in fresh if self.source_admission is None or self.source_admission(p)]
                    if eligible:
                        packet = eligible[-1]
                    elif packet.source_monotonic_seconds-self.last_selected_source > maximum_source_interval:
                        self.cadence_failure_observation = dict(
                            observed_monotonic_seconds=now,
                            last_selected_sequence=self.last_selected,
                            latest_sequence=packet.sequence,
                            latest_source_interval_seconds=packet.source_monotonic_seconds-self.last_selected_source,
                            latest_receipt_age_seconds=now-packet.received_monotonic_seconds,
                            maximum_source_interval_seconds=maximum_source_interval,
                            buffered_packets_in_window=len(in_window),
                            fresh_receipts_in_window=len(fresh),
                        )
                        raise ValueError('No fresh received TCP packet within source cadence envelope')
                    else:
                        packet = None
                if packet is not None and self.source_admission is not None and not self.source_admission(packet):
                    packet = None  # Wait for a genuinely fresh source, never re-date an old parse.
                if packet is not None and packet.sequence != self.last_selected:
                    age = now-packet.received_monotonic_seconds
                    if not 0 <= age <= self.maximum_age:
                        raise TimeoutError('Original TCP receipt is stale/future; never re-date it')
                    self.last_selected = packet.sequence
                    self.last_selected_source = packet.source_monotonic_seconds
                    self.cadence_constrained_selections += packet is not self.latest
                    return packet
                if now >= deadline or self.stopped.is_set():
                    raise TimeoutError('No new TCP source before bounded deadline')
                self.condition.wait(min(.01, deadline-now))

    def close(self):
        self.stopped.set()
        try:
            self.client.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.client.close()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(.5)
        if self.thread.is_alive():
            raise RuntimeError('Owned TCP receiver did not terminate')
