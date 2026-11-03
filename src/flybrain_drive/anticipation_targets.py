"""Offline future teacher-command targets; never actor observations.

Use original labelled decision times within ONE complete episode. Nearest
future labels must lie within an explicit tolerance and the supervision mask.
No interpolation, boundary padding, time shifting or absent-label repair.
"""

import numpy as np


def future_command_targets(times, labels, label_valid, fit_mask,
                           horizons=(.25, .5, 1., 2.), tolerance=.025):
    times = np.asarray(times, dtype=float)
    labels = np.asarray(labels, dtype=float)
    label_valid, fit_mask = np.asarray(label_valid), np.asarray(fit_mask)
    horizons = np.asarray(horizons, dtype=float)
    n = len(times)
    if (times.shape != (n,) or labels.shape != (n, 3)
            or label_valid.shape != (n,) or fit_mask.shape != (n,)
            or label_valid.dtype != np.bool_ or fit_mask.dtype != np.bool_
            or n < 2 or not np.isfinite(times).all()
            or not np.isfinite(labels).all() or np.any(np.diff(times) <= 0)
            or np.any(fit_mask & ~label_valid)
            or horizons.ndim != 1 or len(horizons) == 0
            or not np.isfinite(horizons).all() or np.any(horizons <= 0)
            or np.any(np.diff(horizons) <= 0)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Ordered episode clocks, finite commands, masks and horizons required')
    eligible = np.flatnonzero(label_valid & fit_mask)
    if not len(eligible):
        raise ValueError('Episode has no supervised labels')
    targets = np.zeros((n, len(horizons), 3), dtype=float)
    mask = np.zeros((n, len(horizons)), dtype=bool)
    source_indices = np.full(mask.shape, -1, dtype=np.int64)
    for h, horizon in enumerate(horizons):
        wanted = times + horizon
        right = np.searchsorted(times[eligible], wanted)
        lo, hi = eligible[np.clip(right - 1, 0, len(eligible) - 1)], eligible[np.clip(right, 0, len(eligible) - 1)]
        chosen = np.where(abs(times[lo] - wanted) <= abs(times[hi] - wanted), lo, hi)
        valid = (fit_mask & (chosen > np.arange(n))
                 & (abs(times[chosen] - wanted) <= tolerance)
                 & (wanted <= times[eligible[-1]]))
        # Never bridge an internal unqualified supervision interval.
        gaps = np.cumsum(~fit_mask & label_valid)
        valid &= gaps[chosen] == gaps
        targets[valid, h] = labels[chosen[valid]]
        mask[:, h] = valid
        source_indices[valid, h] = chosen[valid]
    return targets, mask, source_indices


def diagnostic_features(data):
    """Explicit information probes, not deployable policies.

    Body missing values are zero WITH their original masks. Time is an explicitly
    forbidden-actor diagnostic control. No targets or teacher commands in inputs.
    """
    scene = np.asarray(data['scene'], dtype=float)
    body = np.asarray(data['body'], dtype=float)
    valid = np.asarray(data['body_valid'])
    times = np.asarray(data['decision_time'], dtype=float)
    if (scene.shape != (len(times), 576) or body.shape != (len(times), 21)
            or valid.shape != body.shape or valid.dtype != np.bool_
            or not np.isfinite(scene).all() or not np.isfinite(body).all()):
        raise ValueError('Original finite four-view spatial features and body masks required')
    shaped = scene.reshape(-1, 4, 9, 16)
    body_features = np.concatenate([np.where(valid, body, 0), valid], axis=1)
    elapsed = times - times[0]
    time_features = np.column_stack([elapsed, *[elapsed ** p for p in (2, 3, 4)]])
    return dict(body=body_features, vision=scene,
                full=np.concatenate([scene, body_features], axis=1),
                spatial_mean=shaped.mean(-1).reshape(-1, 36),
                intensity=shaped[:, :, :1].reshape(-1, 64),
                motion=shaped[:, :, 1:].reshape(-1, 512),
                time_only=time_features)


def ridge_fit(x, y, regularization=1.):
    """Training-only normalization and fixed ridge strength, with intercept."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or regularization <= 0):
        raise ValueError('Finite matching training matrices required')
    mean, scale, target_mean = x.mean(0), x.std(0), y.mean(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x - mean) / scale
    weights = np.linalg.solve(z.T @ z + regularization * len(x) * np.eye(x.shape[1]),
                              z.T @ (y - target_mean))
    return dict(mean=mean, scale=scale, target_mean=target_mean, weights=weights)


def ridge_predict(model, x):
    return ((x - model['mean']) / model['scale']) @ model['weights'] + model['target_mean']
