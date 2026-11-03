from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.teacher_curriculum import LapEvidence, EvaluationStateGuard, speed_request
from flybrain_drive.ac_teacher_pilot import collection_speed_request


def test_governor_preserves_reviewed_low_speed_behavior_and_brakes_above_target():
    for target in (20, 40, 80):
        for speed in (0, target-2, target, target+8):
            np.testing.assert_allclose(speed_request([.2, .8, -.8], speed, target),
                collection_speed_request([.2, .8, -.8], speed, target))
    assert speed_request([.2, 1, -1], 290, 280)[1] == -1
    with pytest.raises(ValueError):
        speed_request([0, 1, -1], 0, float('nan'))


def timing(lap, last=120000, status='live'):
    return SimpleNamespace(completed_laps=lap, last_time_raw=last, status=status)


def test_partial_lap_discarded_full_lap_and_tyre_exit_retained():
    tracker = LapEvidence()
    for lap in range(3):
        for i in range(100):
            tracker.update(i/100, timing(lap), int(lap == 2 and i == 50))
    tracker.update(0, timing(3), 0)
    assert len(tracker.laps) == 2
    assert tracker.laps[0]['clean_all_tyres']
    assert not tracker.laps[1]['clean_all_tyres']
    assert tracker.laps[0]['official_validity_verified'] is False
    assert tracker.laps[0]['lap_time_ms'] == 120000


def test_teleport_and_lap_reset_never_count_as_completion():
    tracker = LapEvidence()
    tracker.update(.3, timing(3), 0)
    with pytest.raises(ValueError, match='discontinuity'):
        tracker.update(.6, timing(3), 0)
    tracker = LapEvidence()
    tracker.update(.3, timing(3), 0)
    with pytest.raises(ValueError, match='reset'):
        tracker.update(.301, timing(0), 0)


def test_automatic_shift_neutral_has_bounded_coast_and_never_accepts_reverse():
    guard = EvaluationStateGuard()
    state = dict(speed=25., gap=.1, actualGear=2, out_of_track=False)
    assert guard.check(state, 2, 1, 1., 120) is False
    state['actualGear'] = 1
    assert guard.check(state, 1, 1, 1.04, 120) is True
    assert guard.check(state, 1, 1, 1.2, 120) is True
    with pytest.raises(ValueError, match='300ms'):
        guard.check(state, 1, 1, 1.35, 120)
    with pytest.raises(ValueError, match='Unqualified'):
        guard.check(state, 1, 0, 1.2, 120)
    state['actualGear'] = 0
    with pytest.raises(ValueError, match='Reverse'):
        guard.check(state, 0, 1, 1.2, 120)
    state['actualGear'] = 3
    assert guard.check(state, 3, 1, 1.22, 120) is False


def test_reference_gap_is_distinct_from_native_track_exit_in_lap_evaluation():
    state = dict(speed=30., gap=-3.1, actualGear=3, out_of_track=False)
    with pytest.raises(ValueError, match='envelope'):
        EvaluationStateGuard().check(state, 3, 1, 1., 160)
    guard = EvaluationStateGuard(max_gap_m=12.)
    assert guard.check(state, 3, 1, 1., 160) is False
    state['out_of_track'] = True
    with pytest.raises(ValueError, match='envelope'):
        guard.check(state, 3, 1, 1.04, 160)
    state['out_of_track'], state['gap'] = False, 12.01
    with pytest.raises(ValueError, match='envelope'):
        guard.check(state, 3, 1, 1.08, 160)


def test_hotlap_timer_start_without_counter_increase_anchors_first_full_lap():
    tracker = LapEvidence()
    tracker.update(.99, SimpleNamespace(completed_laps=0, last_time_raw=0,
        current_time_raw=10000, status='live'), 0)
    tracker.update(0., SimpleNamespace(completed_laps=0, last_time_raw=0,
        current_time_raw=10, status='live'), 0)
    for i in range(1, 100):
        tracker.update(i/100, SimpleNamespace(completed_laps=0, last_time_raw=0,
            current_time_raw=i*1000, status='live'), 0)
    tracker.update(0., SimpleNamespace(completed_laps=1, last_time_raw=100000,
        current_time_raw=10, status='live'), 0)
    assert len(tracker.laps) == 1 and tracker.laps[0]['lap_time_ms'] == 100000
    assert tracker.laps[0]['spline_distance_laps'] == pytest.approx(1.)


def test_native_timer_reset_away_from_finish_never_anchors_a_lap():
    tracker = LapEvidence()
    tracker.update(.5, SimpleNamespace(completed_laps=0, last_time_raw=0,
        current_time_raw=10000, status='live'), 0)
    with pytest.raises(ValueError, match='timer reset'):
        tracker.update(.501, SimpleNamespace(completed_laps=0, last_time_raw=0,
            current_time_raw=10, status='live'), 0)


def delayed_finish_tracker():
    tracker=LapEvidence()
    for i in range(100):tracker.update(i/100,timing(0),0)
    tracker.update(0,timing(1),0)
    for i in range(1,100):tracker.update(i/100,timing(1),0)
    tracker.previous_time=113187
    tracker.update(.0002587,SimpleNamespace(completed_laps=1,last_time_raw=113227,
        current_time_raw=6,status='live',received_monotonic_seconds=140.),0)
    return tracker


def test_real_native_timer_before_counter_edge_counts_nothing_until_confirmed():
    tracker=delayed_finish_tracker()
    assert tracker.laps==[] and tracker.pending_finish is not None
    tracker.update(.0008,SimpleNamespace(completed_laps=2,last_time_raw=113227,
        current_time_raw=46,status='live',received_monotonic_seconds=140.04),0)
    assert len(tracker.laps)==1 and tracker.laps[0]['lap_time_ms']==113227
    assert tracker.pending_finish is None
    assert tracker.delayed_counter_finishes[0]['lag_seconds']==pytest.approx(.04)


@pytest.mark.parametrize('delay,last',[(.081,113227),(.04,113228)])
def test_late_or_changed_finish_counter_never_confirms_a_lap(delay,last):
    tracker=delayed_finish_tracker()
    with pytest.raises(ValueError,match='Unconfirmed'):
        tracker.update(.0008,SimpleNamespace(completed_laps=2,last_time_raw=last,
            current_time_raw=46,status='live',received_monotonic_seconds=140.+delay),0)
    assert tracker.laps==[]
