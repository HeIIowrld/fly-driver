import json
import struct

import numpy as np
import pytest

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.virtual_body import VirtualBodyEncoder, decode_virtual_body_prefix

IDENTITY = dict(car_model="bmw_z4_gt3", shared_memory_version="1.7", ac_version="1.16.4")
CONTRACT = "configs/ac-virtual-body-v1.json"


def sample(packet=1, receipt=10.0, pitch=0.02, roll=-0.03, heading=1.2):
    raw = bytearray(308)
    struct.pack_into("<i", raw, 0, packet)
    struct.pack_into("<4f", raw, 72, 1000, 2000, 3000, 4000)
    struct.pack_into("<4f", raw, 152, 60, 61, 62, 63)
    struct.pack_into("<3f", raw, 208, heading, pitch, roll)
    struct.pack_into("<3f", raw, 296, 0, 0, 0)
    return decode_virtual_body_prefix(bytes(raw), receipt), bytes(raw)


def test_static_tilt_is_not_angular_rate_and_heading_does_not_reach_actor():
    encoder = VirtualBodyEncoder(CONTRACT, native_identity=IDENTITY)
    native, raw = sample()
    body = encoder.encode(native, episode_id="a", decision_monotonic_seconds=10.01)
    np.testing.assert_allclose(body.values[:2], [0.02, 0.03])
    assert body.values[2:8].tolist() == [0] * 6
    assert body.valid[2:8].tolist() == [False, False, True, False, False, False]
    np.testing.assert_allclose(body.values[8:12], [0.1, 0.2, 0.3, 0.4])
    assert not any("heading" in name or "position" in name for name in body.feature_names)
    assert not {"pitch", "roll", "heading"} & decode_physics_prefix(raw, 10.0).fields.keys()
    alternative, _ = sample(packet=2, receipt=10.02, heading=-2.0)
    later = encoder.encode(alternative, episode_id="a", decision_monotonic_seconds=10.03)
    np.testing.assert_array_equal(body.values, later.values)
    assert not body.values.flags.writeable and not body.valid.flags.writeable


def test_packet_reuse_cannot_refresh_sensor_age():
    encoder = VirtualBodyEncoder(CONTRACT, native_identity=IDENTITY)
    native, _ = sample()
    encoder.encode(native, episode_id="a", decision_monotonic_seconds=10.01)
    repeated, _ = sample(receipt=10.06)
    stale = encoder.encode(repeated, episode_id="a", decision_monotonic_seconds=10.07)
    assert stale.original_received_monotonic_seconds == 10.0
    assert not stale.valid.any() and not stale.values.any()


def test_unregistered_native_acceleration_never_reaches_body_features():
    encoder = VirtualBodyEncoder(CONTRACT, native_identity=IDENTITY)
    _, raw = sample(packet=1)
    first = bytearray(raw)
    struct.pack_into('<3f', first, 44, 0.06, -0.01, 0.57)
    first_sample = decode_virtual_body_prefix(bytes(first), 10.0)
    first_body = encoder.encode(first_sample, episode_id='a', decision_monotonic_seconds=10.01)

    second = bytearray(first)
    struct.pack_into('<i', second, 0, 2)
    struct.pack_into('<3f', second, 44, -0.02, -0.03, -1.15)
    second_sample = decode_virtual_body_prefix(bytes(second), 10.02)
    second_body = encoder.encode(second_sample, episode_id='a', decision_monotonic_seconds=10.03)

    assert first_sample.acceleration_native_g != second_sample.acceleration_native_g
    assert not first_body.valid[5:8].any() and not second_body.valid[5:8].any()
    np.testing.assert_array_equal(first_body.values, second_body.values)
    np.testing.assert_array_equal(first_body.valid, second_body.valid)


def test_packet_reset_requires_episode_boundary_and_same_id_requires_same_bytes():
    encoder = VirtualBodyEncoder(CONTRACT, native_identity=IDENTITY)
    native, _ = sample(packet=2)
    encoder.encode(native, episode_id="a", decision_monotonic_seconds=10.01)
    changed, _ = sample(packet=2, pitch=0.1)
    with pytest.raises(ValueError, match="different bytes"):
        encoder.encode(changed, episode_id="a", decision_monotonic_seconds=10.02)
    reset, _ = sample(packet=1, receipt=10.02)
    with pytest.raises(ValueError, match="episode boundary"):
        encoder.encode(reset, episode_id="a", decision_monotonic_seconds=10.03)
    assert encoder.encode(reset, episode_id="b", decision_monotonic_seconds=10.03).valid.any()


def test_failed_geometry_axis_cannot_be_enabled_by_editing_mask(tmp_path):
    with open(CONTRACT, encoding="utf8") as stream:
        contract = json.load(stream)
    contract["angular_valid"][0] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf8")
    with pytest.raises(ValueError, match="masks differ"):
        VirtualBodyEncoder(path, native_identity=IDENTITY)
    with pytest.raises(ValueError, match="identity differs"):
        VirtualBodyEncoder(CONTRACT, native_identity=dict(IDENTITY, car_model="different"))
