import json
import mmap
import os
import socket
import struct
import threading
import uuid

import numpy as np
import pytest

from flybrain_drive.ac_image_buffer import ExistingNamedMapping, decode_image_buffer
from flybrain_drive.ac_telemetry import ChannelCalibration, LapTimeClock
from flybrain_drive.ac_transport import TelemetryClient, decode_telemetry, encode_control_request
from flybrain_drive.sensor_contract import SampleTime


def packet(step=0, **fields):
    wire = json.dumps({"steps": step, "timestamp_ac": 500., **fields}).encode()
    return decode_telemetry(wire, 1.)


@pytest.mark.parametrize("bad", [b"__import__('os').environ", b'{"steps":0,"steps":1}',
    b'{"steps":0,"timestamp_ac":NaN}', b'{"steps":0,"timestamp_ac":1,"x":1e400}',
    b'{"steps":0,"timestamp_ac":1,"x":9007199254740992}',
    b'{"steps":true,"timestamp_ac":1}', b'[]', b'x'*65508],
    ids=["expression", "duplicate", "nan", "infinity", "large_int", "bool_seq", "array", "size"])
def test_transport_rejects_non_json_or_ambiguous_or_nonfinite_packets(bad):
    with pytest.raises(ValueError):
        decode_telemetry(bad, 1.)


def test_packet_keeps_source_and_receipt_clocks_separate_and_raw_data_immutable():
    data = packet(wheels=[1, 2])
    assert data.source_monotonic_seconds == 500 and data.received_monotonic_seconds == 1
    with pytest.raises(TypeError):
        data.raw["wheels"][0] = 7
    assert json.loads(data.wire_bytes)["wheels"] == [1, 2]


def test_calibration_never_uses_pose_or_progress_as_a_body_sensor():
    with pytest.raises(ValueError, match="[Pp]ermitted"):
        ChannelCalibration("angular_velocity", ("NormalizedSplinePosition",), [[1], [1], [1]], "x")
    data = packet(angular_velocity_x=1, angular_velocity_y=2, angular_velocity_z=3,
                  world_position_x=900)
    model = ChannelCalibration("angular_velocity", ("angular_velocity_x", "angular_velocity_y",
        "angular_velocity_z"), [[0, 0, 2], [1, 0, 0], [0, -1, 0]], "synthetic-axis-fixture")
    result = model.convert(data, SampleTime(.01, 1))
    with pytest.raises(ValueError, match="receipt"):
        model.convert(data, SampleTime(.01, 2))
    np.testing.assert_array_equal(result.values, [6, 1, -2])
    assert result.unit == "rad/s" and result.frame == "body"
    missing = model.convert(packet(angular_velocity_x=1), SampleTime(.01, 1))
    np.testing.assert_array_equal(missing.valid, [False, True, False])
    np.testing.assert_array_equal(missing.values, [0, 1, 0])


def test_unobserved_acceleration_axis_is_invalid_not_fabricated():
    profile = ChannelCalibration("specific_force", ("accelX", "accelY"),
                                 [[9.81, 0], [0, 9.81], [0, 0]], "synthetic-only")
    result = profile.convert(packet(accelX=1, accelY=-1), SampleTime(0, 1))
    np.testing.assert_array_equal(result.valid, [True, True, False])
    np.testing.assert_allclose(result.values, [9.81, -9.81, 0])


def test_instrumented_tyre_temperature_and_load_are_allowed_body_feedback():
    corners = ("fl", "fr", "rl", "rr")
    temperatures = tuple(s + "_tire_temperature_core" for s in corners)
    loads = tuple(s + "_wheel_load" for s in corners)
    fields = dict(zip(temperatures, [-20, 60, 90, 110], strict=True))
    fields.update(dict(zip(loads, [100, 0, 300, -1], strict=True)))
    data = packet(**fields)
    thermal = ChannelCalibration("tyre_core_temperature", temperatures, np.eye(4),
        "explicit-Celsius-fixture-not-native-unit-verification", offset=[273.15] * 4)
    load = ChannelCalibration("wheel_normal_load", loads, np.eye(4), "explicit-newton-fixture")
    result = thermal.convert(data, SampleTime(.1, 1))
    np.testing.assert_allclose(result.values, [253.15, 333.15, 363.15, 383.15])
    assert result.unit == "K" and result.valid.all()
    result = load.convert(data, SampleTime(.1, 1))
    np.testing.assert_array_equal(result.valid, [True, True, True, False])
    np.testing.assert_allclose(result.values, [100, 0, 300, 0])
    missing = thermal.convert(packet(), SampleTime(.1, 1))
    assert not missing.valid.any() and not missing.values.any()


def test_calibration_rejects_invalid_offset_and_scalar_matrix():
    with pytest.raises(ValueError):
        ChannelCalibration("wheel_normal_load", ("fl_wheel_load",), 1., "fixture")
    with pytest.raises(ValueError):
        ChannelCalibration("wheel_normal_load", ("fl_wheel_load",), [[1]] * 4,
                           "fixture", offset=[1, 2])


def test_lap_transition_preserves_time_and_reset_cannot_silently_clear_state():
    clock = LapTimeClock(.001)
    assert clock.update(packet(0, completedLaps=3, currentTime=900)) == .9
    assert clock.update(packet(1, completedLaps=3, currentTime=950)) == pytest.approx(.95)
    with pytest.raises(ValueError, match="Reset|backwards"):
        clock.update(packet(2, completedLaps=0, currentTime=1))
    assert clock.update(packet(3, completedLaps=4, currentTime=10, iLastTime=1000)) == 1.01
    with pytest.raises(ValueError, match="Clock packet"):
        clock.update(packet(3, completedLaps=4, currentTime=20))


def test_image_layout_keeps_black_pixels_and_does_not_invent_exposure_identity():
    image = bytes([5, 10, 20, 0, 0, 0])
    raw = struct.pack("<I", 1) + b"\xff"*6 + image
    decoded = decode_image_buffer(raw, 2, 1, "bgr")
    np.testing.assert_array_equal(decoded.pixels, [[[20, 10, 5], [0, 0, 0]]])
    assert decoded.exposure_id is None and decoded.exposure_time is None
    assert not decoded.synchronization_verified and not decoded.pixels.flags.writeable
    with pytest.raises(ValueError, match="index"):
        decode_image_buffer(struct.pack("<I", 3)+raw[4:], 2, 1, "bgr")
    with pytest.raises(ValueError, match="size"):
        decode_image_buffer(raw[:-1], 2, 1, "bgr")


def test_control_codec_matches_upstream_ranges_without_claiming_applied_control():
    request = json.loads(encode_control_request(.2, 0, 1, 17, steering_sign=-1))
    assert request["steer"] == -.2 and request["acc"] == -1 and request["brake"] == 1
    assert request["server_steps"] == 17
    with pytest.raises(ValueError):
        encode_control_request(0, 2, 0, 0, steering_sign=1)


def test_real_udp_loopback_handshake_and_reordering_without_any_actuator_message():
    producer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    producer.bind(("127.0.0.1", 0))
    producer.settimeout(1)
    messages = []
    errors = []

    def server():
        try:
            data, peer = producer.recvfrom(1024)
            messages.append(data)
            producer.sendto(b"identified", peer)
            for sequence in (0, 2, 1):
                producer.sendto(json.dumps({"steps": sequence, "timestamp_ac": 3.}).encode(), peer)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=server)
    worker.start()
    try:
        with TelemetryClient("127.0.0.1", producer.getsockname()[1]) as client:
            client.subscribe()
            assert client.receive().sequence == 0
            assert client.receive().sequence == 2
            with pytest.raises(ValueError, match="reordered"):
                client.receive()
    finally:
        worker.join(2)
        producer.close()
    assert not worker.is_alive() and not errors and messages == [b"connect"]


@pytest.mark.skipif(os.name != "nt", reason="Windows native named mapping")
def test_missing_named_map_is_not_created_by_probe():
    name = "Local\\FlybrainMissing-" + uuid.uuid4().hex
    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            ExistingNamedMapping(name, 4)


@pytest.mark.skipif(os.name != "nt", reason="Windows native named mapping")
def test_native_windows_mapping_reads_owned_fixture_and_closes_cleanly():
    name = "Local\\FlybrainFixture-" + uuid.uuid4().hex
    payload = struct.pack("<I", 0) + bytes([1, 2, 3, 0]) + b"\xff"*4
    with mmap.mmap(-1, len(payload), tagname=name) as writer:
        writer.write(payload)
        with ExistingNamedMapping(name, len(payload)) as reader:
            np.testing.assert_array_equal(decode_image_buffer(reader.read(), 2, 2, "gray").pixels,
                                          [[1, 2], [3, 0]])
        with pytest.raises(RuntimeError, match="closed"):
            reader.read()
