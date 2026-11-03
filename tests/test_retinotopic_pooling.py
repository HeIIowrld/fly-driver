import numpy as np

from flybrain_drive.retinotopic_pooling import window_pool_weights


def test_window_never_reads_outside_columns_and_preserves_constant_activity():
    x, y = np.meshgrid(np.linspace(.05, .95, 13), np.linspace(.05, .95, 13))
    uv = np.column_stack([x.ravel(), y.ravel()])
    weight = window_pool_weights(uv, bounds=[0, 1, .4, .67])
    assert (weight[:, (uv[:, 1] < .4) | (uv[:, 1] > .67)] == 0).all()
    np.testing.assert_allclose(weight.sum(1), 1, atol=1e-7)
    np.testing.assert_allclose(weight@np.ones(len(uv)), 1, atol=1e-7)
    mirrored = window_pool_weights(np.column_stack([1-uv[:, 0], uv[:, 1]]),
                                   bounds=[0, 1, .4, .67])
    np.testing.assert_allclose(weight.reshape(4, 4, -1),
                               mirrored.reshape(4, 4, -1)[:, ::-1], atol=1e-7)
