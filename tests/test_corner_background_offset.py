"""Ensure optional background offsets preserve the pinned default and reject invalid pins."""
from pathlib import Path

import numpy as np
import pytest

from scripts.infer_frozen_next_corner import load_readout
from scripts.train_vision_teacher_line import read


PINS = Path("runs/vision-frozen-next-corner-20261110-01/pins.json")


def test_background_offset_changes_only_background_score():
    pins = read(PINS)
    default, threshold = load_readout(pins)
    explicit_zero, zero_threshold = load_readout(dict(pins, background_score_offset=0.))
    candidate, candidate_threshold = load_readout(dict(pins, background_score_offset=-.02))
    features = np.zeros((2, 1536), dtype=np.float32)
    base = default(features, features, return_scores=True)
    zero = explicit_zero(features, features, return_scores=True)
    changed = candidate(features, features, return_scores=True)
    assert threshold == zero_threshold == candidate_threshold
    for actual, expected in zip(zero, base, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(changed[0][:, 0], base[0][:, 0] - .02, atol=1e-15)
    np.testing.assert_array_equal(changed[0][:, 1:], base[0][:, 1:])
    np.testing.assert_array_equal(changed[1], base[1])
    np.testing.assert_array_equal(changed[2], base[2])


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -float("inf"), True, "0.02", None])
def test_background_offset_requires_finite_numeric_pin(offset):
    with pytest.raises(ValueError, match="finite number"):
        load_readout(dict(read(PINS), background_score_offset=offset))

def test_batch_readout_matches_live_single_decision_arithmetic():
    pins = read(PINS)
    predict, threshold = load_readout(pins)
    source = Path("runs/vision-monza-two-lap-presence-20261111-01/features.npz")
    with np.load(source, allow_pickle=False) as packed:
        observed = packed["reflected_available"]
        direct = packed["direct"][observed]
        mirror = packed["reflected"][observed]
    batch = predict(direct, mirror, return_scores=True)
    rows = [predict(direct[i:i+1], mirror[i:i+1], return_scores=True)
            for i in range(len(direct))]
    for field, actual in enumerate(batch):
        expected = np.concatenate([row[field] for row in rows])
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(batch[2] >= threshold,
                                  np.array([row[2][0] >= threshold for row in rows]))
