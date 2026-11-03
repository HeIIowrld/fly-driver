import math

import pytest

from scripts.prepare_ac_steering_domain import (
    CALIBRATION_MAGNITUDES,
    diagnostic_piecewise_preimage,
    signed_sweep,
    validation_magnitudes,
)


def test_whole_sweep_covers_both_endpoints_and_two_directions_without_large_jump():
    rows = signed_sweep(CALIBRATION_MAGNITUDES)
    assert len(rows) == 45
    assert [r["index"] for r in rows] == list(range(45))
    values = [r["steering"] for r in rows]
    assert min(values) == -1 and max(values) == 1
    assert values[0] == values[-1] == 0
    assert max(abs(b - a) for a, b in zip(values, values[1:], strict=False)) <= 0.10000000000000009
    for magnitude in CALIBRATION_MAGNITUDES[:-1]:
        assert values.count(magnitude) == values.count(-magnitude) == 2


def test_midpoints_are_registered_before_measurement_and_not_fitted_knots():
    points = validation_magnitudes()
    assert len(points) == 11
    assert not set(points).intersection(CALIBRATION_MAGNITUDES)
    assert points[0] == 0.025 and points[-1] == 0.95
    assert len(signed_sweep(points)) == 45


@pytest.mark.parametrize("points", [(), (0.1, 0.1), (0.2, 0.1), (0.0,), (1.1,), (math.nan,)])
def test_bad_sweep_cannot_expand_or_reorder_a_registered_protocol(points):
    with pytest.raises(ValueError):
        signed_sweep(points)


def test_saturation_preserves_all_preimages_and_never_chooses_optimal_label():
    xs, ys = (-1.0, -0.5, 0.0, 0.5, 1.0), (-1.0, -1.0, 0.0, 1.0, 1.0)
    assert diagnostic_piecewise_preimage(xs, ys, 1.0) == ((0.5, 1.0),)
    assert diagnostic_piecewise_preimage(xs, ys, -1.0) == ((-1.0, -0.5),)
    assert diagnostic_piecewise_preimage(xs, ys, 0.0) == ((0.0, 0.0),)
    assert diagnostic_piecewise_preimage(xs, ys, 0.5) == ((0.25, 0.25),)


def test_unreachable_target_is_explicit_and_not_clipped_or_extrapolated():
    assert diagnostic_piecewise_preimage((-0.1, 0.1), (-0.14, 0.14), 0.8) == ()


@pytest.mark.parametrize(
    "xs,ys,t",
    [
        ((-1.0, 0.0, 1.0), (-1.0, 0.5, 0.4), 0.0),
        ((0.0, 0.0), (0.0, 1.0), 0.5),
        ((-1.0, 1.0), (-1.0, 1.0), math.nan),
        ((-1.0, 1.0), (-1.0, math.inf), 0.0),
        ((-2.0, 1.0), (-1.0, 1.0), 0.0),
    ],
)
def test_nonmonotonic_or_invalid_map_cannot_supply_inverse_labels(xs, ys, t):
    with pytest.raises(ValueError):
        diagnostic_piecewise_preimage(xs, ys, t)
