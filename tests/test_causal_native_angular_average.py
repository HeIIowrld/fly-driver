import numpy as np
import pytest
from flybrain_drive.causal_native_angular_average import integrate_received_angular_velocity


def test_irregular_linear_samples_are_integrated_in_native_order():
    times = np.array([0., .02, .05, .08])
    values = np.column_stack([times, 2*times+1, -times])
    result = integrate_received_angular_velocity(times, values, start=.01, end=.07, decision=.08)
    np.testing.assert_allclose(result, [.04, 1.08, -.04], atol=1e-12)


def test_no_future_bracket_or_extrapolation():
    times = [0., .02, .04, .06]
    values = np.ones((4, 3))
    with pytest.raises(ValueError, match='bracket'):
        integrate_received_angular_velocity(times, values, start=.01, end=.05, decision=.05)
    with pytest.raises(ValueError, match='bracket'):
        integrate_received_angular_velocity(times, values, start=-.01, end=.03, decision=.04)


def test_rejects_duplicate_receipts_and_large_gaps():
    with pytest.raises(ValueError, match='ordered'):
        integrate_received_angular_velocity([0., .02, .02], np.ones((3, 3)), start=0., end=.02, decision=.02)
    with pytest.raises(ValueError, match='gap'):
        integrate_received_angular_velocity([0., .02, .09], np.ones((3, 3)), start=.01, end=.08, decision=.09)
