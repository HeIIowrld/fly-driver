"""Measure camera support and admit a bend when a noisy second edge cannot resolve its sign."""

import numpy as np


def supported_scores(by_camera, visibility):
    """Require the candidate and its two neighbors to be visible in one camera."""
    score = np.full_like(by_camera[0], -np.inf)
    for current, visible in zip(by_camera, visibility, strict=True):
        supported = np.zeros_like(visible, dtype=bool)
        supported[:, 1:-1] = visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:]
        np.maximum(score, np.where(supported, current, -np.inf), out=score)
    return score


def signed_windows(stations, sections):
    """Fit each edge and estimate curvature uncertainty from its residuals."""
    result = {}
    valid = np.isfinite(sections).all(axis=1)
    for center in np.arange(4., 22.):
        offsets = stations - center
        keep = valid & (abs(offsets) <= 6.)
        x = offsets[keep]
        if len(x) < 5 or x[0] >= 0 or x[-1] <= 0 or np.max(np.diff(x)) > 4.:
            continue
        design = np.column_stack((np.ones(len(x)), x, x*x))
        inverse = np.linalg.pinv(design)
        y = sections[keep]
        coefficients = inverse @ y
        residual = y - design @ coefficients
        # A 0.15m localization floor avoids infinite confidence on an exact fit.
        noise = np.maximum(.15, np.sqrt(np.sum(residual**2, axis=0) / (len(x) - 3)))
        scale = 2. / (1. + coefficients[1]**2)**1.5
        curvature = scale * coefficients[2]
        sigma = scale * np.linalg.norm(inverse[2]) * noise
        result[float(center)] = (curvature, sigma)
    return result


def curvature_with_uncertainty(stations, sections, cut=.002):
    """Use one signed edge only if the other edge gives no strong contrary evidence."""
    result = []
    for station, (k, sigma) in signed_windows(stations, sections).items():
        lower = np.maximum(abs(k) - 1.64 * sigma, 0.)
        if k[0] * k[1] > 0 and min(abs(k)) >= cut:
            combined = np.copysign(min(abs(k)), k[0])
        elif np.count_nonzero(lower >= cut) == 1:
            winner = int(np.argmax(lower))
            combined = np.copysign(lower[winner], k[winner])
        else:
            combined = 0.
        result.append((station, float(combined)))
    return result
