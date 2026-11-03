import json
import struct
import numpy as np
import pytest
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.native_angular_history import NativeAngularHistory
from flybrain_drive.view_yaw_interval import prepare_view_yaw_interval
from flybrain_drive.virtual_body import VirtualBodyEncoder


def sensor(packet, time, yaw):
    raw = bytearray(PHYSICS_PREFIX.size)
    struct.pack_into('<i', raw, 0, packet)
    struct.pack_into('<3f', raw, 296, 9., yaw, -7.)
    return decode_physics_prefix(bytes(raw), time)


def encoder():
    identity = json.load(open('configs/ac-virtual-body-v1.json', encoding='utf8'))['native_identity']
    return VirtualBodyEncoder('configs/ac-virtual-body-v1.json', native_identity=identity)


def test_registered_yaw_only_and_causal_brackets():
    h = NativeAngularHistory()
    for i, t in enumerate((10., 10.02, 10.04)):
        h.offer(sensor(i, t, .25))
    e = encoder()
    result = prepare_view_yaw_interval(h, e, previous_render_lower=10.005,
                                       current_render_lower=10.035, camera_receipt=10.04,
                                       decision=10.04)
    assert result.yaw_left_mean_rad_s == pytest.approx(.25)
    assert result.yaw_left_integral_rad == pytest.approx(.0075)
    assert (result.first_packet, result.last_packet) == (0, 2)
    assert not result.physical_time_verified and not result.actor_qualified
    assert not hasattr(result, 'roll') and not hasattr(result, 'pitch')


def test_missing_future_and_bad_time_are_rejected():
    h = NativeAngularHistory()
    h.offer(sensor(1, 10., .2))
    h.offer(sensor(2, 10.04, .2))
    e = encoder()
    with pytest.raises(ValueError, match='bracket'):
        prepare_view_yaw_interval(h, e, previous_render_lower=10., current_render_lower=10.03,
                                  camera_receipt=10.03, decision=10.03)
    with pytest.raises(ValueError, match='ordered'):
        prepare_view_yaw_interval(h, e, previous_render_lower=10.02, current_render_lower=10.01,
                                  camera_receipt=10.04, decision=10.04)


def test_yaw_contract_mask_cannot_be_bypassed():
    h = NativeAngularHistory()
    h.offer(sensor(1, 10., .2)); h.offer(sensor(2, 10.04, .2))
    e = encoder(); e.angular_valid[2] = False
    with pytest.raises(ValueError, match='Validated yaw'):
        prepare_view_yaw_interval(h, e, previous_render_lower=10., current_render_lower=10.03,
                                  camera_receipt=10.04, decision=10.04)
