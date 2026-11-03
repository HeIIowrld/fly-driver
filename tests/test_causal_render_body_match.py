import pytest

from flybrain_drive.causal_render_body_match import (
    select_prior_body_receipt,
    select_prior_body_receipts_per_view,
)


def test_uses_latest_receipt_proven_before_every_view():
    match = select_prior_body_receipt(
        [9.97, 9.995, 10.002], [10.0, 10.001, 10.002, 10.003], [10.004, 10.005, 10.006, 10.007]
    )
    assert match.native_index == 1
    assert match.maximum_host_age_seconds == pytest.approx(0.012)
    assert match.physical_synchronization_verified is False


def test_rejects_future_missing_and_too_old_receipts():
    assert select_prior_body_receipt([], [1] * 4, [1.01] * 4).reason == "no_receipts"
    assert select_prior_body_receipt([1.005], [1] * 4, [1.01] * 4).reason == (
        "no_proven_prior_receipt"
    )
    stale = select_prior_body_receipt([0.9], [1] * 4, [1.01] * 4)
    assert stale.native_index is None and stale.reason == "receipt_too_old"


def test_equal_timestamp_is_not_proof_of_prior_receipt():
    match = select_prior_body_receipt([0.99, 1.0], [1.0] * 4, [1.01] * 4)
    assert match.native_index == 0
    assert select_prior_body_receipt([1.0], [1.0] * 4, [1.01] * 4).reason == (
        "no_proven_prior_receipt"
    )


def test_rejects_invalid_bounds_or_nonmonotonic_receipts():
    with pytest.raises(ValueError):
        select_prior_body_receipt([1, 0.9], [1] * 4, [1.01] * 4)
    with pytest.raises(ValueError):
        select_prior_body_receipt([0.9], [1] * 4, [0.99] * 4)
    with pytest.raises(ValueError):
        select_prior_body_receipt([0.9], [1] * 4, [1.01] * 4, maximum_age_seconds=0.1)


def test_per_view_uses_strict_prior_and_explicit_stale_mask():
    matches = select_prior_body_receipts_per_view(
        [1.00, 1.02, 1.04], [1.01, 1.02, 1.03, 1.05], 1.06
    )
    assert [m.native_index for m in matches] == [None, None, 1, 2]
    assert [m.reason for m in matches[:2]] == ['receipt_too_old', 'receipt_too_old']
