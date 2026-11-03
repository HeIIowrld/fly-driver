import numpy as np
import pytest

from flybrain_drive.flight_actuator import canonical_controls, stationary_steering_request


def test_motor_intent_signs_and_signed_pedal_contract_are_explicit():
    np.testing.assert_array_equal(canonical_controls([.2, .3]), [-.2, .3, 0.])
    np.testing.assert_array_equal(canonical_controls([-.2, -.3]), [.2, 0., .3])


def test_stationary_limits_do_not_relabel_raw_neural_output_or_enable_pedals():
    response = stationary_steering_request([.8, .9])
    assert response['raw_intent'] == [.8, .9]
    assert response['canonical_control'] == [-.8, .9, 0.]
    assert response['requested_control'] == dict(steering=-.1, throttle=0., brake=1.)
    assert response['unclipped_scaled_steering'] == -.48 and response['steering_limited']
    assert stationary_steering_request([-.05, -.5])['requested_control'] == dict(
        steering=.03, throttle=0., brake=1.)


@pytest.mark.parametrize('intent', [[2., 0.], [0., float('nan')], [0.], [0., 0., 0.]])
def test_invalid_neural_output_never_becomes_an_axis_request(intent):
    with pytest.raises(ValueError):
        stationary_steering_request(intent)


def test_stationary_mode_cannot_silently_loosen_its_actuation_limits():
    with pytest.raises(ValueError):
        stationary_steering_request([0., 0.], maximum_steering=.2)
