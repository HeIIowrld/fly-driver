import copy
import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.ac_pixel_retina import NativePixelRetina


def retina():
    return NativePixelRetina(json.loads(Path('configs/ac-csp-yaw-retina.json').read_text(
        encoding='utf8'))['native_retina'])


def fields(yaw):
    return dict(suspensionTravel=[.02]*4, tyreCoreTemperature=[30]*4,
                wheelLoad=[3000]*4, wheelAngularSpeed=[0]*4, localAngularVel=[.7, yaw, -.8])


@pytest.mark.parametrize('yaw', [-.2, 0., .2])
def test_only_reviewed_yaw_signed_pair_is_enabled_and_old_organs_unchanged(yaw):
    r = retina()
    old_config = copy.deepcopy(r.config)
    del old_config['yaw_rotation_organs']
    old = NativePixelRetina(old_config)
    raw = fields(yaw)
    expected, expected_valid = old._organs({k: v for k, v in raw.items()
                                          if k != 'localAngularVel'}, 10., 10.04)
    values, valid = r._organs(raw, 10., 10.04)
    np.testing.assert_array_equal(values[:126], expected[:126])
    np.testing.assert_array_equal(valid[:126], expected_valid[:126])
    np.testing.assert_allclose(values[126:128], np.tanh(np.maximum([yaw, -yaw], 0)/.1))
    assert valid[126:128].all() and not valid[128:].any()
    assert not values[128:].any()
    assert r.calibration != old.calibration
    assert 'localAngularVel' in r.required_native_fields


def test_yaw_missing_invalid_stale_and_source_reuse_are_explicit():
    r = retina()
    raw = fields(.1)
    r._organs(raw, 10., 10.04)
    with pytest.raises(ValueError, match='Reused angular'):
        r._organs(fields(.2), 10., 10.05)
    invalid = fields(float('nan'))
    values, valid = retina()._organs(invalid, 10., 10.04)
    assert not valid[126:].any() and not values[126:].any()
    del raw['localAngularVel']
    values, valid = retina()._organs(raw, 10., 10.04)
    assert not valid[126:].any()
    values, valid = retina()._organs(fields(.1), 10., 10.3)
    assert not valid.any() and not values.any()
    raw['heading'] = .5
    with pytest.raises(ValueError, match='Only declared'):
        retina()._organs(raw, 10., 10.04)
