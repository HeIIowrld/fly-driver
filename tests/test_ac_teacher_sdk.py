"""Teacher SDK vector conventions and strict separation from actor calibration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_telemetry import ChannelCalibration


def app():
    path = (Path(__file__).resolve().parents[1] / 'integrations/ac/flybrain_telemetry'
            / 'flybrain_telemetry.py')
    spec = importlib.util.spec_from_file_location('teacher_sdk_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_sdk_matches_reviewed_vector_order_and_marks_missing_channels():
    values = dict(WorldPosition=[11, 22, 33], LocalVelocity=[4, 5, 6], AccG=[.1, .2, .3],
                  SlipAngle=[1, 2, 3, 4], Gear=3, RPM=5000, LastFF=.4,
                  NormalizedSplinePosition=.7, LapInvalidated=0)
    ac = SimpleNamespace(getCarState=lambda car, key: values[key])
    acsys = SimpleNamespace(CS=SimpleNamespace(**{key: key for key in values}))
    result = app().read_privileged_teacher(ac, acsys)
    assert result['actor_input'] is False
    row = result['raw']
    assert [row[k] for k in ['world_position_x', 'world_position_y', 'world_position_z']] == [
        33, 11, 22]
    assert [row[k] for k in ['local_velocity_x', 'local_velocity_y', 'local_velocity_z']] == [
        6, 4, 5]
    assert row['accelX'] == .1 and row['accelY'] == .3 and row['SlipAngle_rr'] == 4
    assert row['RPM'] == 5000 and row['actualGear'] == 3
    del values['SlipAngle']
    missing = app().read_privileged_teacher(ac, acsys)['raw']
    assert all(missing['SlipAngle_' + side] is None for side in ['fl', 'fr', 'rl', 'rr'])
    with pytest.raises(ValueError, match='Permitted'):
        ChannelCalibration('steering_angle', ('world_position_x',), np.ones((1, 1)), 'test')
