import numpy as np
import pytest

from flybrain_drive.actuator_codec import project_pedals


def test_projection_preserves_steering_and_representable_controls():
    for pedal in (0., .02, .025, .1, .5, 1.):
        row = project_pedals([-.123, pedal, pedal])
        assert row['encoded_request'] == [-.123, pedal, pedal]
        assert row['squared_projection_distance'] == 0
        assert row['game_application_verified'] is False


def test_gap_projection_reports_raw_error_and_tie_selects_zero():
    for value, expected in ((.005, 0.), (.01, 0.), (.011, .02), (.0165, .02), (.019, .02)):
        row = project_pedals([.1, 0., value])
        assert row['canonical_request'] == [.1, 0., value]
        assert row['encoded_request'] == [.1, 0., expected]
        assert row['pedal_in_gap'] == [False, True]
        assert row['encoding_error'][2] == expected-value
        assert row['squared_projection_distance'] == pytest.approx((expected-value)**2)


def test_projection_is_idempotent_and_nearest_over_whole_gap():
    for value in np.linspace(0., .021, 211):
        row = project_pedals([0., float(value), 0.])
        encoded = row['encoded_request']
        assert project_pedals(encoded)['encoded_request'] == encoded
        assert abs(encoded[1]-value) <= min(value, abs(.02-value)) + 1e-15


@pytest.mark.parametrize('canonical', ([0, -.001, 0], [0, 0, 1.001], [1.01, 0, 0],
                                       [0, float('nan'), 0], [0, 0]))
def test_invalid_controls_are_rejected_without_clipping(canonical):
    with pytest.raises(ValueError):
        project_pedals(canonical)
