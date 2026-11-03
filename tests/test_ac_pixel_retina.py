"""Pixel-space host pairing: masks, duplicate frames, time domains and truth exclusion."""

import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.ac_pixel_retina import NativePixelRetina


def frontend():
    config = json.loads(Path('configs/ac-native-pixel-retina.json').read_text())
    config.update(image_shape=[8, 12], receptors_per_view=12, columns=4,
                  acceptance_sigma_pixels=.5)
    config['regions']['front'] = [0, 0, 12, 8]
    return NativePixelRetina(config)


def fields():
    return dict(suspensionTravel=[.01]*4, tyreCoreTemperature=[80.]*4, wheelLoad=[2000.]*4)


def test_first_duplicate_new_and_stale_frames_preserve_explicit_missing_views_and_organs():
    front = frontend()
    black, white = np.zeros((8, 12, 3), np.uint8), np.full((8, 12, 3), 255, np.uint8)
    first = front.pack(black, capture_id=0, callback_time=.1, decision_time=.11,
                       native_scalars=fields(), native_receipt=.105)
    assert first['view_valid'][0].all() and not first['view_valid'][1:].any()
    assert not first['contrast_valid'].any()
    assert first['organ_valid'].sum() == 84
    assert not first['organ_valid'][:18].any() and not first['organ_valid'][126:].any()
    assert first['organs'][57] == 1  # 80C ->353.15K population center, declared conversion.
    repeated = front.pack(black, capture_id=0, callback_time=.1, decision_time=.13,
                          native_scalars=fields(), native_receipt=.105)
    assert not repeated['contrast_valid'].any()
    new = front.pack(white, capture_id=1, callback_time=.14, decision_time=.15,
                     native_scalars=fields(), native_receipt=.145)
    assert new['contrast_valid'][0].all() and (new['view_contrast'][0] > 0).all()
    assert new['organ_valid'].sum() == 108
    stale = front.pack(white, capture_id=1, callback_time=.14, decision_time=.4,
                       native_scalars=fields(), native_receipt=.145)
    assert not stale['view_valid'].any() and not stale['organ_valid'].any()
    returned = front.pack(black, capture_id=2, callback_time=.41, decision_time=.42,
                          native_scalars=fields(), native_receipt=.415)
    assert returned['view_valid'][0].all() and not returned['contrast_valid'].any()
    assert returned['organ_valid'].sum() == 84
    assert front.calibration['clock_domain'] == 'host_monotonic_receipt'
    assert not front.calibration['body_ray_registration_verified']


def test_privileged_body_keys_changed_frame_identity_and_bad_clock_do_not_advance_history():
    front = frontend()
    image = np.zeros((8, 12, 3), np.uint8)
    with pytest.raises(ValueError, match='Only declared'):
        front.pack(image, capture_id=0, callback_time=.1, decision_time=.11,
                   native_scalars=dict(fields(), world_position_x=10), native_receipt=.105)
    assert front.last_decision is None and not front.visual_history and not front.scalar_history
    front.pack(image, capture_id=0, callback_time=.1, decision_time=.11)
    with pytest.raises(ValueError, match='identity'):
        front.pack(np.ones_like(image), capture_id=0, callback_time=.1, decision_time=.12)
    with pytest.raises(ValueError, match='host order'):
        front.pack(image, capture_id=1, callback_time=.2, decision_time=.15)
    assert front.last_decision == .11


def wheel_frontend():
    config = frontend().config
    config['wheel_rotation_organs'] = json.loads(Path(
        'configs/ac-native-wheel-retina.json').read_text(encoding='utf8'))['wheel_rotation_organs']
    return NativePixelRetina(config)


def test_wheel_rotation_changes_only_reserved_organ_slots_and_preserves_stationary_vs_missing():
    image = np.zeros((8, 12, 3), np.uint8)
    old, new = frontend(), wheel_frontend()
    baseline = old.pack(image, capture_id=0, callback_time=.1, decision_time=.11,
                        native_scalars=fields(), native_receipt=.105)
    stopped = new.pack(image, capture_id=0, callback_time=.1, decision_time=.11,
        native_scalars=dict(fields(), wheelAngularSpeed=[0.]*4), native_receipt=.105)
    assert stopped['organ_valid'].sum() == 98
    assert stopped['organ_valid'][:18].reshape(2, 9)[:, :7].all()
    assert not stopped['organs'][:18].any()
    for key in baseline:
        if key not in ('organs', 'organ_valid'):
            np.testing.assert_array_equal(stopped[key], baseline[key])
    np.testing.assert_array_equal(stopped['organs'][18:], baseline['organs'][18:])
    moving = new.pack(image, capture_id=1, callback_time=.14, decision_time=.15,
        native_scalars=dict(fields(), wheelAngularSpeed=[10., 20., 10., 20.]), native_receipt=.145)
    assert moving['organ_valid'].sum() == 126
    assert moving['organs'][9] > moving['organs'][0] > 0
    missing = new.pack(image, capture_id=2, callback_time=.18, decision_time=.19,
                       native_scalars=fields(), native_receipt=.185)
    assert not missing['organ_valid'][:18].any() and not missing['organs'][:18].any()
    assert (old.calibration['organ_calibration_sha256']
            != new.calibration['organ_calibration_sha256'])


def test_wheel_invalid_pair_duplicate_identity_and_stale_reset_are_explicit():
    new = wheel_frontend()
    image = np.zeros((8, 12, 3), np.uint8)
    f = dict(fields(), wheelAngularSpeed=[10., 20., float('nan'), 20.])
    first = new.pack(image, capture_id=0, callback_time=.1, decision_time=.11,
                     native_scalars=f, native_receipt=.105)
    assert not first['organ_valid'][:9].any() and first['organ_valid'][9:16].all()
    with pytest.raises(ValueError, match='Reused wheel'):
        new.pack(image, capture_id=0, callback_time=.1, decision_time=.12,
            native_scalars=dict(fields(), wheelAngularSpeed=[5., 20., 5., 20.]),
            native_receipt=.105)
    assert new.last_decision == .11
    stale = new.pack(image, capture_id=1, callback_time=.2, decision_time=.4,
                     native_scalars=f, native_receipt=.105)
    assert not stale['organ_valid'].any()
    returned = new.pack(image, capture_id=2, callback_time=.41, decision_time=.42,
        native_scalars=dict(fields(), wheelAngularSpeed=[10.]*4), native_receipt=.415)
    assert returned['organ_valid'][:18].reshape(2, 9)[:, :7].all()
    assert not returned['organ_valid'][:18].reshape(2, 9)[:, 7:].any()
    with pytest.raises(ValueError, match='Reused wheel'):
        new.pack(image, capture_id=2, callback_time=.41, decision_time=.43,
            native_scalars=dict(fields(), wheelAngularSpeed=[5., 10., 15., 10.]),
            native_receipt=.415)
    assert new.last_decision == .42
