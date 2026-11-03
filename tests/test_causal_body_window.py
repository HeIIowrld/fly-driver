import pytest

from flybrain_drive.causal_body_window import prior_window_mean


def test_uses_only_prior_held_values_and_excludes_tied_and_future_packets():
    result = prior_window_mean([.92, .96, 1., 1.01], [1., 3., 999., 999.],
                               end_lower=1., end_upper=1., duration=.08)
    assert result.reason == "ok"
    assert result.value == pytest.approx(2.)
    assert not result.physical_synchronization_verified


def test_rejects_missing_history_and_stale_or_gapped_data():
    assert prior_window_mean([.95], [1], end_lower=1., end_upper=1., duration=.1).reason == (
        "insufficient_prior_coverage"
    )
    assert prior_window_mean([.8, .9], [1, 2], end_lower=1., end_upper=1., duration=.1).reason == (
        "receipt_gap_too_large"
    )
    with pytest.raises(ValueError):
        prior_window_mean([.9, .8], [1, 2], end_lower=1., end_upper=1., duration=.1)


def test_last_sample_must_be_fresh_at_latest_render_bound():
    result = prior_window_mean([.92, .96], [1., 3.], end_lower=1., end_upper=1.02,
                               duration=.08)
    assert result.reason == "receipt_gap_too_large"
    assert result.value is None
