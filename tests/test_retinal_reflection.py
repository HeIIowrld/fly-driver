import numpy as np
import pytest

from flybrain_drive.retinal_reflection import (
    reflect_retinal_history,
    reflection_permutation,
)


def hex_uv():
    qr = np.array([(q, r) for q in range(-15, 16) for r in range(-15, 16)
                   if abs(q+r) <= 15])
    x, y = qr[:, 0]+qr[:, 1]/2, qr[:, 1]*np.sqrt(3)/2
    return np.column_stack([.5+x/40, .5+y/40])


def test_registered_reflection_is_exact_involution_and_swaps_side_views():
    uv = hex_uv()
    permutation = reflection_permutation(uv)
    np.testing.assert_allclose(uv[permutation, 0], 1-uv[:, 0], atol=1e-14)
    original = np.arange(2*4*721).reshape(2, 4, 721)
    reflected = reflect_retinal_history(original, permutation)
    np.testing.assert_array_equal(reflected[:, 1], original[:, 2, permutation])
    np.testing.assert_array_equal(reflected[:, 2], original[:, 1, permutation])
    np.testing.assert_array_equal(reflect_retinal_history(reflected, permutation), original)
    np.testing.assert_array_equal(original, np.arange(2*4*721).reshape(2, 4, 721))


def test_non_symmetric_registration_is_not_repaired_with_nearest_neighbors():
    uv = hex_uv()
    uv[0, 0] += .001
    with pytest.raises(ValueError, match='Exact'):
        reflection_permutation(uv)


def test_nonfinite_history_is_rejected():
    original = np.zeros((2, 4, 721))
    original[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        reflect_retinal_history(original, reflection_permutation(hex_uv()))
