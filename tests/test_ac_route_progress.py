"""Synthetic counter/geometry checks; these do not certify an actual full lap."""

import math

import pytest

from flybrain_drive.ac_route_progress import RouteProgress


def row(time=0., position=.99, count=0, x=0., y=0., length=1000.):
    return dict(track_length_m=length, lap_count=count, normalized_spline_position=position,
                native_received_monotonic_seconds=time, x=x, y=y)


def test_counter_wrap_alone_is_only_a_partial_lap_from_an_arbitrary_start():
    progress = RouteProgress(row(), maximum_speed_kmh=40.)
    for index in range(1, 11):
        unwrapped = .99+index*.001
        progress.observe(row(index*.1, unwrapped % 1, int(unwrapped), x=float(index)))
    assert progress.distance_m == pytest.approx(10.)
    assert progress.distance_m < progress.length


@pytest.mark.parametrize('changed', [dict(lap_count=-1), dict(lap_count=1),
    dict(normalized_spline_position=.1), dict(track_length_m=1001.),
    dict(native_received_monotonic_seconds=.3), dict(native_received_monotonic_seconds=0.),
    dict(normalized_spline_position=None), dict(x=100.)])
def test_counter_resets_teleports_unknown_values_and_gaps_reject_without_committing(changed):
    progress = RouteProgress(row(), maximum_speed_kmh=40.)
    invalid = row(.1, .991, x=1.)
    invalid.update(changed)
    with pytest.raises(ValueError):
        progress.observe(invalid)
    assert progress.distance_m == 0.
    assert progress.observe(row(.1, .991, x=1.)) == pytest.approx(1.)


def test_small_fake_progress_cannot_accumulate_with_a_stationary_world_position():
    progress = RouteProgress(row(position=.1), maximum_speed_kmh=40.)
    with pytest.raises(ValueError, match='Implausible'):
        for index in range(1, 100):
            progress.observe(row(index*.1, .1+index*.0001))


def test_positive_lap_counter_reset_is_not_a_new_forward_lap():
    progress = RouteProgress(row(count=2), maximum_speed_kmh=40.)
    with pytest.raises(ValueError, match='counter'):
        progress.observe(row(.1, .991, count=0, x=1.))


def test_curve_progress_is_distinct_from_start_to_current_displacement():
    progress = RouteProgress(row(position=.1), maximum_speed_kmh=40.)
    for index in range(1, 1601):
        arc = index*.3
        angle = arc/200.
        last = row(index*.04, .1+arc/1000., x=200*math.sin(angle),
                   y=200*(1-math.cos(angle)))
        progress.observe(last)
    assert progress.distance_m == pytest.approx(480.)
    assert math.hypot(last['x'], last['y']) < 400.
