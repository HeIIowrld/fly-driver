import struct

import numpy as np
import pytest

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.per_view_body_history import PerViewBodyHistory

IDENTITY = dict(car_model='bmw_z4_gt3', shared_memory_version='1.7', ac_version='1.16.4')


def sample(packet, when, *, speed=36):
    data = bytearray(308)
    struct.pack_into('<i', data, 0, packet)
    struct.pack_into('<f', data, 28, speed)
    struct.pack_into('<4f', data, 72, 1000, 2000, 3000, 4000)
    struct.pack_into('<4f', data, 104, 1, 2, 3, 4)
    struct.pack_into('<4f', data, 152, 60, 61, 62, 63)
    return decode_physics_prefix(bytes(data), when)


def build(h, bounds, camera, decision):
    return h.build(render_start_lowers=tuple(bounds), camera_receipt_time=camera,
                   decision_time=decision)


def test_separate_view_and_current_receipts_and_first_packet_is_missing():
    h = PerViewBodyHistory(episode_id='test', native_identity=IDENTITY)
    h.ingest_native_physics(sample(1, 1.0))
    assert build(h, [1.001]*4, 1.002, 1.003) is None
    h.ingest_native_physics(sample(2, 1.010))
    h.ingest_native_physics(sample(3, 1.020))
    # An already buffered future packet is never selected at a prior decision.
    h.ingest_native_physics(sample(4, 1.040))
    c = build(h, [1.010, 1.011, 1.020, 1.021], 1.025, 1.030)
    assert [x.packet_id if x else None for x in c.visual_contexts] == [None, 2, 2, 3]
    assert c.decision_body.packet_id == 3
    assert c.decision_body.decision_monotonic_seconds == 1.030
    assert c.visual_contexts[1].decision_monotonic_seconds == 1.025


def test_duplicate_is_not_redated_and_stale_data_remain_masked():
    h = PerViewBodyHistory(episode_id='test', native_identity=IDENTITY)
    h.ingest_native_physics(sample(1, 1.0))
    h.ingest_native_physics(sample(2, 1.01))
    assert not h.ingest_native_physics(sample(2, 1.07))
    c = build(h, [1.075]*4, 1.08, 1.081)
    assert c.decision_body.original_received_monotonic_seconds == 1.01
    assert not c.decision_body.valid.any()
    assert all(not x.valid.any() for x in c.visual_contexts)
    with pytest.raises(ValueError, match='changed bytes'):
        h.ingest_native_physics(sample(2, 1.09, speed=37))
    with pytest.raises(ValueError, match='reset'):
        h.ingest_native_physics(sample(1, 1.10))


def test_eviction_is_explicit_missing_and_failed_build_does_not_advance():
    h = PerViewBodyHistory(episode_id='test', native_identity=IDENTITY, capacity=2)
    for i in range(1, 5):
        h.ingest_native_physics(sample(i, 2+i*.01))
    with pytest.raises(ValueError):
        build(h, [2.01]*4, 2.03, 2.02)
    c = build(h, [2.025, 2.035, 2.035, 2.045], 2.05, 2.051)
    assert c.visual_contexts[0] is None
    assert [x.packet_id if x else None for x in c.visual_contexts] == [None, 3, 3, 4]
    assert c.decision_body.packet_id == 4
    assert np.all(c.decision_body.valid[[0, 1, 4, 16]])
    with pytest.raises(ValueError, match='Advancing'):
        build(h, [2.03, 2.04, 2.04, 2.05], 2.0509, 2.051)
