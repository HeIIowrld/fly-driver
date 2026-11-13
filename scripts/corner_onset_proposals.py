"""Supervise both next-corner directions at every distance on a vehicle-relative grid."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DISTANCES = np.arange(4., 21., 2.)
STATIONS = np.arange(2., 25., 2.)


def geometry_features(sections):
    """Represent both edges, widths and signed curvature with explicit visibility."""
    edges = np.asarray(sections, dtype=float)
    if edges.ndim != 3 or edges.shape[1:] != (len(STATIONS), 2):
        raise ValueError("Expected vehicle-relative right and left boundaries at 2–24 m")
    visible = np.isfinite(edges)
    width = edges[:, :, 1] - edges[:, :, 0]
    width_ok = (width > 3.) & (width < 20.)
    valid = visible & (~visible.all(axis=-1) | width_ok)[:, :, None]
    edges = np.where(valid, edges, np.nan)

    # Each direction sees its own inside edge first. Reflection swaps the two rows.
    mirrored = np.stack((edges, -edges[:, :, ::-1]), axis=1)
    signals = [mirrored, np.diff(mirrored, axis=2)]
    for spacing in (1, 2, 3):
        left = mirrored[:, :, :-2 * spacing]
        middle = mirrored[:, :, spacing:-spacing]
        right = mirrored[:, :, 2 * spacing:]
        signals.append((right - 2 * middle + left) / (2. * spacing)**2)

    values = np.concatenate([s.reshape(len(edges), 2, -1) for s in signals], axis=-1)
    return np.concatenate((np.nan_to_num(values, nan=0.), np.isfinite(values)), axis=-1)


def motion_features(original, reflected, valid):
    """Keep ordered times, cameras and T4/T5 channels, pooling each retina to 2×2."""
    branches = []
    for values in (original, reflected):
        grid = np.asarray(values).reshape(-1, 3, 4, 8, 2, 2, 2, 2)
        pooled = grid.mean(axis=(5, 7))
        # Remove global per-channel activity to preserve the retinotopic contrast.
        pooled -= pooled.mean(axis=(-2, -1), keepdims=True)
        branches.append(np.where(valid[:, None], pooled.reshape(len(grid), -1), 0.))
    return np.stack(branches, axis=1)


def targets(rows):
    """Label every direction-distance cell, including frames with no old proposal."""
    y = np.zeros((len(rows), 2, len(DISTANCES)))
    for i, row in enumerate(rows):
        if row["truth"] != "approach":
            continue
        distances = np.asarray(row["truth_endpoints_m"])
        error = np.max(abs(DISTANCES[:, None] - distances), axis=1)
        y[i, int(row["label"]) - 1] = np.exp(-.5 * (error / 2.5)**2)
    return y


def fit(features, labels, weights):
    """Fit a common readout for both mirrored directions using all labeled frames."""
    x = features.reshape(-1, features.shape[-1])
    y = labels.reshape(-1, len(DISTANCES))
    w = np.repeat(weights, 2)
    w *= len(w) / w.sum()
    scale = StandardScaler().fit(x, sample_weight=w)
    model = Ridge(alpha=240., solver="cholesky")
    model.fit(scale.transform(x).clip(-6., 6.), y, sample_weight=w)
    return scale, model


def predict(features, fitted):
    scale, model = fitted
    flat = features.reshape(-1, features.shape[-1])
    return model.predict(scale.transform(flat).clip(-6., 6.)).reshape(-1, 2, len(DISTANCES))


def candidates(scores):
    """Return the strongest distance per direction and abstain on an exact side tie."""
    best = np.argmax(scores, axis=-1)
    peak = np.take_along_axis(scores, best[..., None], axis=-1)[..., 0]
    winner = np.argmax(peak, axis=-1)
    indices = np.arange(len(scores))
    confidence = np.where(peak[:, 0] == peak[:, 1], -np.inf, peak[indices, winner])
    return winner + 1, DISTANCES[best[indices, winner]], confidence
