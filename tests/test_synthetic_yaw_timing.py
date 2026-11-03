import pytest

from flybrain_drive.synthetic_yaw_timing import synthetic_body_at, synthetic_interval_yaw


def test_stationary_moving_and_crossing_intervals():
    args = dict(stationary_seconds=.4, yaw_rate=.12)
    assert synthetic_interval_yaw(.2, .4, **args) == 0
    assert synthetic_interval_yaw(.4, .44, **args) == pytest.approx(.12)
    assert synthetic_interval_yaw(.38, .42, **args) == pytest.approx(.06)
    assert synthetic_interval_yaw(.38, .42, stationary_seconds=.4, yaw_rate=-.12) == pytest.approx(-.06)


@pytest.mark.parametrize('start,end', [(0., 0.), (.2, .1), (-.1, .1), (0., float('nan'))])
def test_invalid_times_are_rejected(start, end):
    with pytest.raises(ValueError):
        synthetic_interval_yaw(start, end, stationary_seconds=.4, yaw_rate=.12)


def test_body_observations_use_their_own_receipt_times():
    args = dict(stationary_seconds=.4, yaw_rate=-.12, forward_speed_m_s=3., wheel_radius_m=.3)
    prior = synthetic_body_at(.39, **args)
    current = synthetic_body_at(.43, **args)
    assert not prior.any()
    assert current[4] == -.12 and current[16] == 3.
    assert current[17:21].tolist() == [10.] * 4
    assert synthetic_body_at(.4, **args)[16] == 0.
    with pytest.raises(ValueError):
        synthetic_body_at(.43, **{**args, 'wheel_radius_m': 0.})
