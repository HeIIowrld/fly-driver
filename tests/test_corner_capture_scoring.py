"""Regression coverage for missing decisions and capture-gap accounting."""
import pytest

from scripts.score_corner_captures import capture_row, forecast_state, summarize


def capture(frame, forecast=None, truth_distance=10., received=None):
    return capture_row(
        "recording", dict(capture_id=frame, received=frame*.1 if received is None else received,
                          generation="generation"),
        forecast, (("1", truth_distance, 3), ("1", truth_distance-.2, 3)), (8., 8.),
    )


LEFT = dict(available=True, alert=True, direction=1, distance_m=12.)


def test_all_captures_remain_in_denominator():
    rows = [capture(0), capture(1, dict(available=False)),
            capture(2, dict(available=True, alert=False, direction=0)), capture(3, LEFT)]
    report = summarize(rows)
    assert report["frames"] == report["approach_frames"] == 4
    assert report["outputs"] == dict(no_output=1, abstain=1, straight=1, left=1, right=0)
    assert report["by_direction"]["left"]["direction_hits"] == 1


def test_missing_decision_breaks_stability_but_keeps_approach():
    rows = [capture(0, LEFT), capture(1), capture(2, LEFT), capture(3, LEFT)]
    report = summarize(rows)
    assert report["approaches"] == 1
    assert report["events"][0]["first_stable_frame"] == 3
    assert report["distance_change"]["pairs"] == 1


def test_distance_hit_requires_both_exposure_endpoints():
    assert capture(0, dict(LEFT, distance_m=14.8))["joint_hit"]
    assert not capture(0, dict(LEFT, distance_m=15.1))["joint_hit"]


def test_false_alert_streak_breaks_at_clock_gap_and_generation():
    rows = [capture(0, LEFT, 22), capture(1, LEFT, 22), capture(2, LEFT, 22, received=.7),
            capture(3, LEFT, 22, received=.8)]
    rows[-1]["generation"] = "next"
    report = summarize(rows)["false_alerts"]["outside_band"]
    assert report["episodes"] == 3
    assert report["longest_frames"] == 2


@pytest.mark.parametrize("distance", [None, float("nan"), float("inf"), -1., 21.])
def test_alert_requires_finite_supported_distance(distance):
    with pytest.raises(ValueError):
        forecast_state(dict(LEFT, distance_m=distance))
