"""Fit an offline racing-line teacher inside installed road boundaries.

The teacher output is a low-curvature reference under a nominal vehicle model.
Track coordinates are used here solely to generate and evaluate targets.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

MARGIN_M = 1.1
GRID_M = 2.0
REGULARIZATION = 0.0002


def resample_loop(points, arc, stations):
    points = np.r_[points, points[:1]]
    return np.column_stack([np.interp(stations, arc, axis) for axis in points.T])


def periodic_difference(size):
    return (diags((1, -2, 1), (-1, 0, 1), shape=(size, size), format="csr")
            + diags((1, 1), (1-size, size-1), shape=(size, size), format="csr"))


def optimize_line(edges):
    left, right = (edge[:, [2, 0]] for edge in edges)
    center = (left + right) * 0.5
    chord = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    if not np.isfinite(chord).all() or (chord <= 0.).any():
        raise ValueError("Track station coordinates must be finite and distinct")
    arc = np.r_[0., np.cumsum(chord)]
    stations = np.linspace(0., arc[-1], round(arc[-1] / GRID_M), endpoint=False)
    left, right = (resample_loop(edge, arc, stations) for edge in (left, right))
    midpoint = (left + right) * 0.5
    width = np.linalg.norm(left-right, axis=1)
    if (width < 2. * MARGIN_M).any():
        raise ValueError("The configured clearance does not fit the spline widths")
    cross = (left-right) / width[:, None]
    half_width = width * 0.5 - MARGIN_M
    difference = periodic_difference(len(stations))
    # Penalize discrete acceleration and lateral offset on the same center stations.
    reference = difference @ midpoint
    weight = REGULARIZATION

    def objective(offset):
        accel = reference + difference @ (cross * offset[:, None])
        penalty = np.sum(accel**2) + weight * (offset @ offset)
        gradient = 2. * (np.sum(cross * (difference.T @ accel), axis=1) + weight * offset)
        return penalty, gradient

    start = np.zeros(len(stations))
    fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                   bounds=list(zip(-half_width, half_width, strict=True)),
                   options=dict(maxiter=2000, ftol=1e-11, gtol=1e-6))
    if not fit.success or fit.fun > objective(start)[0]:
        raise ValueError(f"Offline line fit did not converge: {fit.message}")
    return midpoint + fit.x[:,None] * cross, midpoint, dict(
        nominal_margin_m=MARGIN_M, station_step_m=float(stations[1]),
        stations=len(stations), objective_center=float(objective(start)[0]),
        objective_fit=float(fit.fun), iterations=int(fit.nit))
