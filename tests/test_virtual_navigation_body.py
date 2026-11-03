import struct

import numpy as np
import pytest

from flybrain_drive.virtual_body import VirtualBodyEncoder, decode_virtual_body_prefix
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)

IDENTITY = dict(car_model="bmw_z4_gt3", shared_memory_version="1.7", ac_version="1.16.4")


def raw(packet=1, speed=36, heading=1.2):
    data = bytearray(308)
    struct.pack_into("<i", data, 0, packet)
    struct.pack_into("<f", data, 28, speed)
    struct.pack_into("<4f", data, 72, 1000, 2000, 3000, 4000)
    struct.pack_into("<4f", data, 104, 1, 2, 3, 4)
    struct.pack_into("<4f", data, 152, 60, 61, 62, 63)
    struct.pack_into("<3f", data, 208, heading, 0.02, -0.03)
    return bytes(data)


def encoder():
    return NavigationBodyEncoder(
        "configs/ac-virtual-navigation-body-v2.json", native_identity=IDENTITY
    )


def test_legacy16_and_masks_are_exact_with_declared_speed_and_wheel_units():
    new = encoder().encode(
        decode_navigation_body_prefix(raw(), 10), episode_id="a", decision_monotonic_seconds=10.01
    )
    old = VirtualBodyEncoder("configs/ac-virtual-body-v1.json", native_identity=IDENTITY).encode(
        decode_virtual_body_prefix(raw(), 10), episode_id="a", decision_monotonic_seconds=10.01
    )
    np.testing.assert_array_equal(new.values[:16], old.values)
    np.testing.assert_array_equal(new.valid[:16], old.valid)
    np.testing.assert_array_equal(new.values[16:], [10, 1, 2, 3, 4])
    assert new.valid[16:].all() and not new.physiological_registration_verified
    assert not any("heading" in n or "position" in n for n in new.feature_names)
    assert not new.values.flags.writeable and not new.valid.flags.writeable


def test_reused_packet_cannot_refresh_either_old_or_new_sensors():
    e = encoder()
    e.encode(
        decode_navigation_body_prefix(raw(), 10), episode_id="a", decision_monotonic_seconds=10.01
    )
    stale = e.encode(
        decode_navigation_body_prefix(raw(), 10.06),
        episode_id="a",
        decision_monotonic_seconds=10.07,
    )
    assert stale.original_received_monotonic_seconds == 10
    assert not stale.valid.any() and not stale.values.any()


def test_changed_packet_bytes_and_negative_speed_are_rejected_without_rerating():
    e = encoder()
    e.encode(
        decode_navigation_body_prefix(raw(), 10), episode_id="a", decision_monotonic_seconds=10.01
    )
    with pytest.raises(ValueError, match="different bytes"):
        e.encode(
            decode_navigation_body_prefix(raw(speed=37), 10.02),
            episode_id="a",
            decision_monotonic_seconds=10.03,
        )
    assert e.legacy.last_decision == 10.01
    with pytest.raises(ValueError, match="negative"):
        decode_navigation_body_prefix(raw(speed=-1), 10)


def test_heading_changes_do_not_reach_navigation_inputs():
    first = encoder().encode(
        decode_navigation_body_prefix(raw(), 10), episode_id="a", decision_monotonic_seconds=10.01
    )
    second = encoder().encode(
        decode_navigation_body_prefix(raw(heading=-2), 10),
        episode_id="a",
        decision_monotonic_seconds=10.01,
    )
    np.testing.assert_array_equal(first.values, second.values)
