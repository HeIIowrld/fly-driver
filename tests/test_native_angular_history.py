import struct
import numpy as np
import pytest
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.native_angular_history import NativeAngularHistory


def sample(packet, when, *, extra=0):
    raw = bytearray(PHYSICS_PREFIX.size)
    struct.pack_into('<i', raw, 0, packet)
    struct.pack_into('<f', raw, 28, float(extra))
    return decode_physics_prefix(bytes(raw), when)


def test_duplicate_never_refreshes_first_receipt_or_window():
    h = NativeAngularHistory()
    one = sample(1, 10.)
    assert h.offer(one)
    assert not h.offer(decode_physics_prefix(one.prefix_bytes, 10.01))
    assert h.offer(sample(2, 10.02))
    w = h.window(start=10., end=10.015, decision=10.02)
    assert (w.first_packet, w.last_packet, w.first_receipt, w.last_receipt) == (1, 2, 10., 10.02)
    assert w.native_mean == (0., 0., 0.) and w.physical_time_verified is False


def test_future_packet_does_not_complete_earlier_window():
    h = NativeAngularHistory()
    h.offer(sample(1, 10.))
    h.offer(sample(2, 10.02))
    h.offer(sample(3, 10.04))
    with pytest.raises(ValueError, match='bracket'):
        h.window(start=10.01, end=10.03, decision=10.03)


def test_changed_duplicate_backward_receipt_and_packet_reset_rejected():
    h = NativeAngularHistory()
    h.offer(sample(2, 10.))
    with pytest.raises(ValueError, match='changed'):
        h.offer(sample(2, 10.01, extra=1))
    assert not h.offer(sample(2, 10.02))
    with pytest.raises(ValueError, match='backwards'):
        h.offer(sample(3, 10.01))
    with pytest.raises(ValueError, match='backwards'):
        h.offer(sample(1, 10.03))
    h.reset()
    assert h.offer(sample(1, 10.03))


def test_eviction_and_large_gap_are_explicit():
    h = NativeAngularHistory(capacity=2)
    for n in range(3):
        h.offer(sample(n, 10. + .02*n))
    with pytest.raises(ValueError, match='bracket'):
        h.window(start=10., end=10.03, decision=10.04)
    h.reset()
    h.offer(sample(1, 10.))
    h.offer(sample(2, 10.08))
    with pytest.raises(ValueError, match='gap'):
        h.window(start=10.01, end=10.07, decision=10.08)
