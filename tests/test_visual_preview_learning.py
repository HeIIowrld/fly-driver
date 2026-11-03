import numpy as np
import pytest

from flybrain_drive.visual_preview_learning import training_moments


def episode(name, role, value, group='original'):
    return dict(id=name, role=role, group=group, x=np.full((3, 576), value, float),
                y=np.full((3, 6), value, float), valid=np.ones((3, 6), bool))


def fit(rows, include=True):
    return training_moments(rows, include_spatial=include, feature_floor=1e-5,
                            target_floor=[.005, .0001]*3)


def test_development_sentinel_cannot_change_any_fit_moment_or_history():
    train = episode('train', 'train', 2)
    expected = fit([train])
    actual = fit([train, episode('held-out', 'development', 1e20)])
    for key in ('feature_mean', 'feature_scale', 'target_mean', 'target_scale'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['fitting_histories'] == ['train']


def test_invalid_endpoints_do_not_change_label_moments_and_spatial_condition_is_explicit():
    old, new = episode('old', 'train', 1), episode('new', 'train', 3, 'new_spatial')
    old['valid'][0] = False
    old['y'][0] = 1e10
    np.testing.assert_array_equal(fit([old, new], include=False)['target_mean'], np.ones(6))
    assert fit([old, new])['fitting_histories'] == ['old', 'new']
    assert fit([old, new])['target_mean'][0] == pytest.approx(2.2)


def test_evaluation_role_is_rejected_instead_of_read_as_training():
    with pytest.raises(ValueError, match='evaluation'):
        fit([episode('evaluation', 'evaluation', 1)])
