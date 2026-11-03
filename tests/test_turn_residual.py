import numpy as np
import pytest

from flybrain_drive.turn_residual import current_turn_prior, residual_weights


def test_body_prior_units_sign_and_unknown_no_target_mask_repair():
    body = np.zeros((4, 21))
    mask = np.ones_like(body, dtype=bool)
    body[:, 4], body[:, 16] = [.2, -.2, .2, .2], [10, 10, 1, 10]
    mask[3, 4] = False
    original = body.copy(), mask.copy()
    prior, available = current_turn_prior(body, mask)
    np.testing.assert_allclose(prior[0], [-.05, -.02, -.1, -.02, -.2, -.02])
    np.testing.assert_array_equal(prior[1], -prior[0])
    assert available.tolist() == [True, True, False, False]
    assert not prior[2:].any()
    np.testing.assert_array_equal(body, original[0])
    np.testing.assert_array_equal(mask, original[1])


def test_weighting_keeps_every_valid_row_and_balances_only_present_classes():
    residual = np.repeat(np.array([[-1.], [0.], [1.], [1.]]), 6, axis=1)
    valid = np.ones_like(residual, dtype=bool)
    valid[0, 1] = False
    weights, counts = residual_weights(residual, valid, np.ones(6), True)
    assert counts[0].tolist() == [1, 1, 2]
    assert counts[1].tolist() == [0, 1, 2]
    np.testing.assert_allclose(weights.sum(0), np.ones(6)/6)
    np.testing.assert_allclose([weights[0, 0], weights[1, 0], weights[2:, 0].sum()], 1/18)
    assert (weights[valid] > 0).all() and not weights[~valid].any()
    uniform, _ = residual_weights(residual, valid, np.ones(6), False)
    np.testing.assert_allclose(uniform[:, 0], np.ones(4)/24)
    with pytest.raises(ValueError):
        residual_weights(residual, valid, np.zeros(6), True)
