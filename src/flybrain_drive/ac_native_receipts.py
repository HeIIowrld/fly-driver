"""Finite, read-only native receipt sampling independent of camera/actor decisions."""

import base64
import hashlib
import json
import threading
import time
from pathlib import Path

from .ac_native_physics import NativePhysicsReader, decode_physics_prefix
from .graph import sha256


def verified_handoff_native_receipts(root, events):
    """Pin independent receipts between the same trial's original native packet anchors."""
    root = Path(root)
    path = root/'expert-handoff/native-receipts.json'
    report = json.loads((root/'expert-handoff/native-receipts-result.json').read_text(
        encoding='utf8'))
    outcome = json.loads((root/'result.json').read_text(encoding='utf8'))
    release = next(e for e in events if e['kind'] == 'handoff_vjoy_released')
    starts = [e for e in events if e['kind'] == 'handoff_native_receipts_started']
    if (report['status'] != 'completed' or report['worker_alive'] or report['error']
            or report['clock'] != 'host_monotonic_QPC_time.perf_counter'
            or report['period_seconds'] != .01 or report['maximum_seconds'] != 40
            or report['maximum_samples'] != 4000 or not 1 <= report['samples'] <= 4000
            or report != outcome.get('independent_native_receipts')
            or report['file_sha256'] != sha256(path) or len(starts) != 1
            or starts[0]['started_at'] != report['started_at']
            or not release['released_at'] <= report['started_at'] < report['stopped_at']
                <= report['started_at']+41):
        raise ValueError('One successful original independent native receipt capture required')
    anchors = [e['native'] for e in events if e['kind'] == 'raw_pair']
    lower = max((r['packet_id'] for r in anchors
                 if r['received_monotonic_seconds'] <= report['started_at']), default=None)
    upper = min((r['packet_id'] for r in anchors
                 if r['received_monotonic_seconds'] >= report['stopped_at']), default=None)
    if lower is None or upper is None or lower >= upper:
        raise ValueError('Same-trial native packet anchors must bracket independent sampling')
    result, previous_time, previous_packet = [], report['started_at'], lower
    for row in json.loads(path.read_text(encoding='utf8')):
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
        if (hashlib.sha256(raw).hexdigest() != row['prefix_sha256']
                or sample.packet_id != row['packet_id']
                or not previous_packet < sample.packet_id <= upper
                or not previous_time < sample.received_monotonic_seconds <= report['stopped_at']):
            raise ValueError('Independent native hash/clock/packet reset or bracketing differs')
        result.append(dict(packet_id=sample.packet_id, fields=dict(sample.fields),
                           received_monotonic_seconds=sample.received_monotonic_seconds))
        previous_time, previous_packet = sample.received_monotonic_seconds, sample.packet_id
    if len(result) != report['samples']:
        raise ValueError('Independent native receipt count differs')
    return result, report


class NativeReceiptCollector:
    """One owned worker; original receipt clocks/bytes only, no interpolation or actuators."""

    def __init__(self, *, reader_factory=None):
        self.reader_factory = reader_factory or (lambda: NativePhysicsReader(
            timeout=.05, poll_seconds=.002, clock=time.perf_counter))
        self.samples, self.error, self.timeouts = [], None, 0
        self.started_at = self.stopped_at = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is not None:
            raise ValueError('Native receipt collector cannot restart')
        self.started_at = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name='ac-native-receipts', daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with self.reader_factory() as reader:
                while not self.stop_event.is_set():
                    if time.perf_counter()-self.started_at >= 40 or len(self.samples) >= 4000:
                        raise TimeoutError('Finite native receipt budget exhausted')
                    try:
                        sample = reader.receive()
                    except TimeoutError:
                        self.timeouts += 1
                        continue
                    if (sample.received_monotonic_seconds < self.started_at or self.samples and (
                            sample.packet_id <= self.samples[-1].packet_id
                            or sample.received_monotonic_seconds
                                <= self.samples[-1].received_monotonic_seconds)):
                        raise ValueError('Native receipt identity or original clock is unordered')
                    self.samples.append(sample)
                    self.stop_event.wait(.01)
        except Exception as error:
            self.error = dict(error=type(error).__name__, reason=str(error))

    def check(self):
        if self.error is not None:
            raise RuntimeError('Independent native receipt capture failed: '+self.error['reason'])

    def close(self):
        if self.thread is None:
            raise ValueError('Native receipt collector was not started')
        self.stop_event.set()
        self.thread.join(timeout=.5)
        if self.thread.is_alive():
            raise TimeoutError('Native receipt worker is still live; do not reuse its output')
        if self.stopped_at is None:
            self.stopped_at = time.perf_counter()
        return dict(status='completed' if self.error is None and self.samples else 'failed',
            started_at=self.started_at, stopped_at=self.stopped_at,
            samples=len(self.samples), period_seconds=.01, maximum_seconds=40,
            maximum_samples=4000, receive_timeouts=self.timeouts, error=self.error,
            worker_alive=False, clock='host_monotonic_QPC_time.perf_counter',
            source_time_verified=False, atomicity_verified=False, actuator_messages_sent=0)
