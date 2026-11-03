"""Explicit engineering attention window on registered columns, not new fly optics."""

import numpy as np


def window_pool_weights(uv, *, bounds, columns=4, rows=4, sigma=.18):
    uv = np.asarray(uv, float)
    x0, x1, y0, y1 = bounds
    if (uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all()
            or not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1 or sigma <= 0):
        raise ValueError('Finite registered coordinates and explicit normalized window required')
    coordinates = (uv-[x0, y0])/[x1-x0, y1-y0]
    x, y = np.meshgrid((np.arange(columns)+.5)/columns, (np.arange(rows)+.5)/rows)
    centers = np.column_stack([x.ravel(), y.ravel()])
    weight = np.exp(-np.sum((centers[:, None]-coordinates[None])**2, axis=-1)/(2*sigma**2))
    inside = ((uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1))
    weight[:, ~inside] = 0
    if not np.all(weight.sum(1) > 0):
        raise ValueError('Every declared window bin requires actual source neurons')
    return (weight/weight.sum(1, keepdims=True)).astype(np.float32)
