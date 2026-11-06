"""Diagnostic decomposition of a fixed retinal pooling operator.

This is engineering linear algebra, not a new biological receptive field.
The discarded component is orthogonal to the row space of the OLD pool.
Neither targets nor episode clocks participate in this decomposition.
"""
import numpy as np


def pool_null_projector(pool):
    pool = np.asarray(pool, dtype=np.float64)
    if (pool.ndim != 2 or not 0 < pool.shape[0] < pool.shape[1]
            or not np.isfinite(pool).all()
            or np.linalg.matrix_rank(pool) != pool.shape[0]):
        raise ValueError('Finite full-row-rank old retinal pooling matrix required')
    return np.linalg.pinv(pool)


def discarded_retinal_component(retina, pool, inverse):
    retina = np.asarray(retina, dtype=np.float64)
    pool, inverse = np.asarray(pool, float), np.asarray(inverse, float)
    if (retina.ndim < 1 or retina.shape[-1] != pool.shape[1]
            or inverse.shape != (pool.shape[1], pool.shape[0])
            or not np.isfinite(retina).all() or not np.isfinite(inverse).all()):
        raise ValueError('Original ordered retinal samples and fixed pooling inverse required')
    pooled = retina @ pool.T
    reconstructed = pooled @ inverse.T
    return retina - reconstructed
