import numpy as np
import pytest
from flybrain_drive.bounded_steering_fit import fit_bounded_steering


def test_fit_uses_six_muscles_and_reduces_real_command_error():
    rng = np.random.default_rng(821)
    x = rng.normal(size=(400, 6))
    y = np.tanh(x @ np.array([.6, -.3, .1, 0, .2, -.1]) + .1)
    result = fit_bounded_steering(x, y, np.ones(len(x)), np.zeros(6), 0.)
    prediction = np.tanh(x @ result['beta'] + result['bias'])
    assert np.mean((prediction-y)**2) < .001 * np.mean(y**2)
    assert result['final_loss'] < result['initial_loss']


def test_feature_units_and_missing_columns_do_not_break_fit():
    rng = np.random.default_rng(822)
    x = rng.normal(size=(100, 6)); x[:, 5] = 2
    y = .5 * np.tanh(x[:, 0])
    a = fit_bounded_steering(x, y, np.ones(100), np.zeros(6), 0.)
    scaled = x * np.array([100, .1, 2, 5, 10, 1])
    b = fit_bounded_steering(scaled, y, np.ones(100), np.zeros(6), 0.)
    np.testing.assert_allclose(x @ a['beta'] + a['bias'], scaled @ b['beta'] + b['bias'], atol=1e-8)


@pytest.mark.parametrize('change', ['weights', 'target', 'features'])
def test_invalid_contract_is_rejected(change):
    x = np.ones((10, 6)); y = np.zeros(10); w = np.ones(10)
    if change == 'weights': w[0] = 0
    if change == 'target': y[0] = 1.1
    if change == 'features': x[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_bounded_steering(x, y, w, np.zeros(6), 0.)
