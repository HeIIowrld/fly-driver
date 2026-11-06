import numpy as np
import pytest

from flybrain_drive.spatial_pool_residual import pool_null_projector, discarded_retinal_component


def test_discarded_signal_is_invisible_to_old_pool():
    pool = np.array([[.5, .5, 0, 0], [0, 0, .5, .5]])
    x = np.array([[1., -1., 2., -2.], [1., 3., 5., 7.]])
    inverse = pool_null_projector(pool)
    residual = discarded_retinal_component(x, pool, inverse)
    np.testing.assert_allclose(residual @ pool.T, 0, atol=1e-14)
    np.testing.assert_allclose(x - residual, (x @ pool.T) @ inverse.T)
    np.testing.assert_allclose(residual[0], x[0])


def test_decomposition_does_not_modify_original_data():
    pool = np.array([[.25, .75, 0], [0, .4, .6]])
    x = np.arange(24.).reshape(2, 4, 3)
    original = x.copy()
    residual = discarded_retinal_component(x, pool, pool_null_projector(pool))
    assert residual.shape == x.shape
    np.testing.assert_array_equal(x, original)


def test_rejects_degenerate_pool_and_wrong_retinal_order_length():
    with pytest.raises(ValueError):
        pool_null_projector(np.ones((2, 4)))
    pool = np.array([[1., 0, 0], [0, 1., 0]])
    with pytest.raises(ValueError):
        discarded_retinal_component(np.ones(4), pool, pool_null_projector(pool))
