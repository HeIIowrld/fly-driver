import base64
import hashlib
import json
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_native_receipts import (
    NativeReceiptCollector,
    verified_handoff_native_receipts,
)
from flybrain_drive.graph import sha256


class Reader:
    def __init__(self, *, reset=False, timeout=False):
        self.packet, self.reset, self.timeout = 0, reset, timeout
        self.closed = False
        self.received = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def receive(self):
        self.packet += 1
        self.received.set()
        if self.timeout:
            raise ValueError('Original mapping disappeared')
        return SimpleNamespace(packet_id=1 if self.reset else self.packet,
                               received_monotonic_seconds=time.perf_counter())


def test_independent_reader_retains_original_samples_and_closes_one_worker():
    reader = Reader()
    capture = NativeReceiptCollector(reader_factory=lambda: reader)
    capture.start()
    assert reader.received.wait(1)
    report = capture.close()
    assert report['status'] == 'completed' and report['samples'] >= 1
    assert report['actuator_messages_sent'] == 0 and reader.closed
    assert not report['worker_alive'] and capture.close() == report
    assert capture.samples[0].received_monotonic_seconds >= report['started_at']
    with pytest.raises(ValueError, match='cannot restart'):
        capture.start()


def test_mapping_failure_cannot_be_reported_as_success():
    reader = Reader(timeout=True)
    capture = NativeReceiptCollector(reader_factory=lambda: reader)
    capture.start()
    capture.thread.join(1)
    with pytest.raises(RuntimeError, match='mapping disappeared'):
        capture.check()
    assert capture.close()['status'] == 'failed' and reader.closed


def test_duplicate_packet_is_never_redated_to_create_a_future_target():
    reader = Reader(reset=True)
    capture = NativeReceiptCollector(reader_factory=lambda: reader)
    capture.start()
    capture.thread.join(1)
    assert not capture.thread.is_alive()
    assert capture.close()['status'] == 'failed'
    assert len(capture.samples) == 1


def test_finite_deadline_stops_capture_without_a_new_game_command():
    capture = NativeReceiptCollector(reader_factory=Reader)
    capture.started_at = time.perf_counter()-41
    capture._run()
    with pytest.raises(RuntimeError, match='budget exhausted'):
        capture.check()
    assert not capture.samples


def receipt_fixture(root, *, mutation=None):
    folder = root/'expert-handoff'
    folder.mkdir()
    rows = []
    for packet, stamp in [(12, 1.2), (13, 1.25), (14, 1.3)]:
        raw = struct.pack('<i', packet)+bytes(304)
        rows.append(dict(packet_id=packet, received_monotonic_seconds=stamp,
                         prefix_base64=base64.b64encode(raw).decode(),
                         prefix_sha256=hashlib.sha256(raw).hexdigest()))
    if mutation:
        mutation(rows)
    path = folder/'native-receipts.json'
    path.write_text(json.dumps(rows), encoding='utf8')
    report = dict(status='completed', worker_alive=False, error=None,
                  clock='host_monotonic_QPC_time.perf_counter', period_seconds=.01,
                  maximum_seconds=40, maximum_samples=4000, samples=3,
                  started_at=1.1, stopped_at=1.4, file_sha256=sha256(path))
    (folder/'native-receipts-result.json').write_text(json.dumps(report), encoding='utf8')
    (root/'result.json').write_text(json.dumps(dict(independent_native_receipts=report)),
                                    encoding='utf8')
    return [dict(kind='handoff_vjoy_released', released_at=1.05),
            dict(kind='handoff_native_receipts_started', started_at=1.1),
            *[dict(kind='raw_pair', native=dict(packet_id=packet,
                 received_monotonic_seconds=stamp)) for packet, stamp in [(10, 1.), (20, 1.5)]]]


def test_verified_receipts_are_original_bytes_with_same_trial_packet_anchors(tmp_path):
    events = receipt_fixture(tmp_path)
    native, report = verified_handoff_native_receipts(tmp_path, events)
    assert [row['packet_id'] for row in native] == [12, 13, 14]
    assert native[0]['received_monotonic_seconds'] == 1.2
    assert native[0]['fields']['gas'] == 0 and report['samples'] == 3


@pytest.mark.parametrize('mutation', [
    lambda rows: rows[1].update(received_monotonic_seconds=1.1),
    lambda rows: rows[1].update(packet_id=12),
    lambda rows: rows[1].update(prefix_sha256='0'*64),
    lambda rows: rows[2].update(received_monotonic_seconds=1.6),
])
def test_independent_capture_cannot_change_hash_clock_or_packet(tmp_path, mutation):
    events = receipt_fixture(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match='hash/clock/packet'):
        verified_handoff_native_receipts(tmp_path, events)


def test_unanchored_receipts_cannot_claim_the_live_game_identity(tmp_path):
    events = receipt_fixture(tmp_path)
    with pytest.raises(ValueError, match='anchors'):
        verified_handoff_native_receipts(tmp_path, events[:-1])
