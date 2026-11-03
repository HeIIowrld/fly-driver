import pytest

from flybrain_drive.flight_moving_probe import moving_component_request


def test_harness_brakes_in_prefix_and_termination_despite_arbitrary_model_intent():
    for phase in ('stationary_prefix', 'braking'):
        request = moving_component_request([1., 1.], phase=phase, speed_kmh=2.)
        assert request['requested_control'] == dict(steering=0., throttle=0., brake=1.)
        assert request['canonical_control'] == [-1., 1., 0.]
        assert request['steering_override']


def test_speed_overshoot_cannot_request_simultaneous_throttle_and_brake():
    request = moving_component_request([-.1, 1.], phase='launch', speed_kmh=8.)
    assert request['requested_control']['brake'] > 0
    assert request['requested_control']['throttle'] == 0
    assert request['requested_control']['steering'] == .06


@pytest.mark.parametrize('speed', [10., -1., float('nan')])
def test_invalid_or_out_of_budget_speed_never_produces_a_control(speed):
    with pytest.raises(ValueError):
        moving_component_request([0., 0.], phase='launch', speed_kmh=speed)
