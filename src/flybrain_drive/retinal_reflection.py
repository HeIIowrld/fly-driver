"""Engineering image reflection at the registered retina, not a neuron-type swap."""

import numpy as np


def reflection_permutation(image_uv):
    uv = np.asarray(image_uv)
    if uv.shape != (721, 2) or not np.isfinite(uv).all():
        raise ValueError('Registered finite721 retinal image positions required')
    reflected = uv.copy()
    reflected[:, 0] = 1-reflected[:, 0]
    squared = ((reflected[:, None]-uv[None])**2).sum(2)
    permutation = squared.argmin(1)
    if (np.max(squared[np.arange(721), permutation]) > 1e-12
            or len(set(permutation.tolist())) != 721
            or not np.array_equal(permutation[permutation], np.arange(721))):
        raise ValueError('Exact bijective involutive retinal reflection required')
    return permutation


def reflect_retinal_history(retina, permutation):
    value = np.asarray(retina)
    if value.ndim != 3 or value.shape[1:] != (4, 721) or not np.isfinite(value).all():
        raise ValueError('Complete finite four-view retinal history required')
    return value[:, [0, 2, 1, 3]][:, :, permutation].copy()
