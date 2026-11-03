"""Engineering current-turn prior, not a fly circuit or measured future road."""

import numpy as np


def current_turn_prior(body, valid):
    """Body yaw-left rad/s and unsigned m/s -> right bearing/curvature at 5/10/20m.

    Constant speed, negligible sideslip and constant curvature are approximations.
    Missing yaw/speed or speed <=1m/s yields an explicit unavailable, zero prior.
    This never changes target validity or removes a recorded observation.
    """
    body, valid = np.asarray(body), np.asarray(valid)
    if (body.ndim != 2 or body.shape[1] != 21 or valid.shape != body.shape
            or valid.dtype != bool or not np.isfinite(body).all()):
        raise ValueError('Finite body21 with explicit original validity required')
    available = valid[:, 4] & valid[:, 16] & (body[:, 16] > 1)
    curvature = np.zeros(len(body))
    curvature[available] = -body[available, 4]/body[available, 16]
    prior = np.empty((len(body), 6))
    for h, distance in enumerate((5., 10., 20.)):
        prior[:, 2*h] = .5*distance*curvature
        prior[:, 2*h+1] = curvature
    return prior, available


def residual_weights(train_residual, valid, scale, balanced):
    """Six equally weighted heads; optional TRAIN-only sign-class balancing.

    Positive/neutral/negative classes use 0.25 original TRAIN target std.
    Every valid row has positive mass; empty classes are not synthesized.
    """
    residual, valid, scale = map(np.asarray, (train_residual, valid, scale))
    if (residual.ndim != 2 or residual.shape[1] != 6 or valid.shape != residual.shape
            or valid.dtype != bool or scale.shape != (6,) or not (scale > 0).all()
            or not np.isfinite(scale).all() or not np.isfinite(residual).all()
            or not valid.any(0).all() or type(balanced) is not bool):
        raise ValueError('Whole finite TRAIN residuals, six valid heads and scales required')
    classes = np.where(residual < -.25*scale, 0, np.where(residual > .25*scale, 2, 1))
    weights = np.zeros_like(residual, dtype=float)
    counts = np.zeros((6, 3), dtype=int)
    for h in range(6):
        counts[h] = np.bincount(classes[valid[:, h], h], minlength=3)
        weights[valid[:, h], h] = 1.
        if balanced:
            weights[valid[:, h], h] /= counts[h, classes[valid[:, h], h]]
        weights[:, h] /= 6*weights[:, h].sum()
    return weights, counts
