import base64
import struct
import time

import pytest

from flybrain_drive.native_process_sampler import NativeProcessSampler


def packet(number=7):
    raw = bytearray(420)
    struct.pack_into('<i', raw, 0, number)
    struct.pack_into('<f', raw, 28, 60.)
    return bytes(raw)


def test_process_delivers_exact_record_and_stops():
    raw = packet()
    sampler = NativeProcessSampler(recorded=(raw,), period=.01, capacity=128)
    sampler.start()
    row = sampler.poll(timeout=3)
    assert row is not None
    assert base64.b64decode(row['authority']['prefix_base64']) == raw
    assert row['authority']['packet_id'] == 7
    assert row['copy']['synthetic_replay'] is True
    sampler.stop()
    assert not sampler.process.is_alive()


def test_stale_sample_is_rejected_not_retimestamped():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.01, capacity=128)
    sampler.start()
    with pytest.raises(RuntimeError, match='stale or future'):
        sampler.poll(timeout=3, clock=lambda: time.perf_counter()+1)
    with pytest.raises(RuntimeError, match='stale or future'):
        sampler.poll(timeout=3)
    sampler.stop()


def test_full_queue_fails_explicitly():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.002, capacity=1)
    sampler.start()
    deadline = time.perf_counter()+3
    while sampler.process.is_alive() and time.perf_counter() < deadline:
        time.sleep(.01)
    with pytest.raises(RuntimeError, match='queue full'):
        sampler.poll(timeout=.2)
    with pytest.raises(RuntimeError, match='queue full'):
        sampler.poll(timeout=.2)
    with pytest.raises(RuntimeError, match='queue full'):
        sampler.stop()


def test_large_full_queue_does_not_keep_failed_producer_alive():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.0001, capacity=128)
    sampler.start()
    sampler.process.join(8)
    assert not sampler.process.is_alive()
    with pytest.raises(RuntimeError, match='queue full'):
        sampler.poll(timeout=.2)
    with pytest.raises(RuntimeError, match='queue full'):
        sampler.stop()


def test_producer_completion_is_not_silently_treated_as_live():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.01, max_samples=1)
    sampler.start()
    assert sampler.poll(timeout=3) is not None
    with pytest.raises(RuntimeError, match='producer terminated'):
        sampler.poll(timeout=3)
    sampler.stop()


def test_unexpected_producer_death_is_explicit():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.01)
    sampler.start()
    assert sampler.poll(timeout=3) is not None
    sampler.process.terminate()
    sampler.process.join(3)
    with pytest.raises(RuntimeError, match='without clean terminal receipt'):
        sampler.stop()


def test_rejects_bad_configuration():
    with pytest.raises(ValueError):
        NativeProcessSampler(recorded=(b'bad',))
    with pytest.raises(ValueError):
        NativeProcessSampler(max_age=.051)
