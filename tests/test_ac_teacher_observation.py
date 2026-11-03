"""Teacher packet assembly and history guards; no live game or actor control."""

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, decode_physics_prefix
from flybrain_drive.ac_sac_teacher import CHANNELS
from flybrain_drive.ac_teacher_observation import (
    PrivilegedTeacherObservationBuilder,
    TeacherInputUnavailable,
)
from flybrain_drive.ac_transport import decode_telemetry


def pair(index, *, session='test', source_seconds=None, skew=.001, extension=True):
    extra = {key: 0. for key in CHANNELS if key not in ('speed', 'gap', 'angular_velocity_y')}
    extra.update(world_position_x=1., world_position_y=2., NormalizedSplinePosition=.1,
                 track_length_m=1000.)
    row = dict(steps=index + 1, timestamp_ac=100 + index * .04 if source_seconds is None
               else source_seconds, producer_session_id=session, speed=1., steerAngle=0.,
               accStatus=.1, brakeStatus=0., angular_velocity_y=0.)
    if extension:
        row['privileged_teacher'] = dict(protocol='flybrain.ac.privileged_teacher.v1',
                                         actor_input=False, raw=extra)
    host = 20 + index * .04
    packet = decode_telemetry(json.dumps(row).encode(), host)
    raw = bytearray(308)
    struct.pack_into('<i', raw, 0, index + 1)
    offset = 0
    for name, _code, width in PHYSICS_LAYOUT:
        if name == 'heading':
            struct.pack_into('<f', raw, offset, -.3)
        offset += width * 4
    return packet, decode_physics_prefix(bytes(raw), host + skew)


def builder():
    geometry = SimpleNamespace(features=lambda x, y, yaw: dict(
        gap=.1, sensors=np.ones(11) * 100))
    curve = SimpleNamespace(segment=lambda distance: np.zeros(12))
    return PrivilegedTeacherObservationBuilder(geometry, curve, profile='legacy_20241008')


def test_builder_waits_for_three_real_history_rows_and_keeps_teacher_state_readonly():
    value = builder()
    for index in range(3):
        result = value.build(*pair(index))
        assert result.observation is None and not result.actor_input
    result = value.build(*pair(3))
    assert result.observation.shape == (125,) and not result.observation.flags.writeable
    assert result.state['LapDist'] == 100 and result.state['out_of_track'] == 0
    assert len(value.history) == 3 and not result.exposure_synchronization_verified
    with pytest.raises(TypeError):
        result.state['gap'] = 100


def test_missing_namespace_bad_cadence_or_session_change_does_not_advance_history():
    value = builder()
    with pytest.raises(TeacherInputUnavailable, match='extension not loaded'):
        value.build(*pair(0, extension=False))
    assert value.previous is None and not value.history
    value.build(*pair(0))
    before = value.previous
    for args, message in [
            (pair(1, source_seconds=100.2), 'cadence'),
            (pair(1, skew=.1), 'skew'),
            (pair(1, session='restarted'), 'session/sequence'),
            (pair(0), 'session/sequence')]:
        with pytest.raises(ValueError, match=message):
            value.build(*args)
        assert value.previous == before and len(value.history) == 1
    assert value.build(*pair(1)).observation is None
