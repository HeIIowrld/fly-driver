import numpy as np
import pytest

from flybrain_drive.anticipation_targets import (diagnostic_features,
    future_command_targets, ridge_fit, ridge_predict)


def episode():
    t = np.arange(101) * .02
    y = np.column_stack([t, t / 2, t / 3])
    return t, y, np.ones(101, bool), np.ones(101, bool)


def test_future_labels_are_original_and_do_not_cross_episode_end():
    t, y, valid, fit = episode()
    target, mask, indices = future_command_targets(t, y, valid, fit, [.5], .001)
    np.testing.assert_array_equal(indices[:76, 0], np.arange(25, 101))
    np.testing.assert_array_equal(target[:76, 0], y[25:])
    assert not mask[76:].any()
    assert np.all(indices[~mask] == -1)


def test_unlabelled_ticks_and_missing_future_are_not_repaired():
    t, y, valid, fit = episode()
    valid[25] = fit[25] = False
    _, mask, _ = future_command_targets(t, y, valid, fit, [.5], .001)
    assert not mask[0, 0]
    assert not mask[25, 0]


def test_internal_disqualified_label_interval_is_not_bridged():
    t, y, valid, fit = episode()
    fit[20] = False
    _, mask, _ = future_command_targets(t, y, valid, fit, [.5], .001)
    assert not mask[:21].any()
    assert mask[21, 0]


def test_rejects_nonadvancing_clocks_and_mask_contradiction():
    t, y, valid, fit = episode()
    t[2] = t[1]
    with pytest.raises(ValueError):
        future_command_targets(t, y, valid, fit)
    t, y, valid, fit = episode()
    valid[3] = False
    with pytest.raises(ValueError):
        future_command_targets(t, y, valid, fit)


def test_diagnostics_never_ingest_labels_and_preserve_masks():
    data = dict(scene=np.ones((3, 576)), body=np.ones((3, 21)),
                body_valid=np.ones((3, 21), bool), decision_time=np.arange(3.))
    data['body_valid'][:, 0] = False
    first = diagnostic_features(data)
    data['labels'] = np.full((3, 3), 99.)
    data['body'][:, 0] = 1e9
    second = diagnostic_features(data)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first['intensity'].shape == (3, 64)
    assert first['motion'].shape == (3, 512)


def test_ridge_normalization_is_fitted_only_on_train():
    x = np.arange(20.).reshape(10, 2)
    y = x[:, :1] * 2
    model = ridge_fit(x, y, .000001)
    mean = model['mean'].copy()
    ridge_predict(model, x + 1000)
    np.testing.assert_array_equal(model['mean'], mean)
    assert np.max(abs(ridge_predict(model, x) - y)) < .001
