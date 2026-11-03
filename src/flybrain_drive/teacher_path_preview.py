"""Privileged future-path labels only; never an actor observation adapter.

Positive bearing is camera screen-right. Curvature is a pure-pursuit chord
surrogate (2*lateral/chord_squared), not differential road curvature or an
optimal racing line. Future distances are horizontal travelled arc lengths.
"""

import numpy as np


def future_path_preview(telemetry, clock, look, right, distances=(5., 10., 20.)):
    """Interpolate within contiguous driving telemetry, without bridging gaps.

    Missing endpoints are explicit invalid auxiliary labels. Original controls,
    clocks, histories and masks must remain untouched by the caller.
    """
    t = np.asarray([r['script_clock'] for r in telemetry], float)
    p = np.asarray([r['position'] for r in telemetry], float)
    distances = np.asarray(distances, float)
    if (len(t) < 2 or p.shape != (len(t), 3) or not np.isfinite(p).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or not np.isfinite(clock) or distances.ndim != 1
            or np.any(distances <= 0) or not np.isfinite(distances).all()):
        raise ValueError('Finite strictly ordered clocks/positions and positive distances required')
    f, r = (np.asarray(v, float)[[0, 2]] for v in (look, right))
    if (not np.isfinite([f, r]).all() or min(np.linalg.norm(f), np.linalg.norm(r)) < .9):
        raise ValueError('Near-horizontal finite camera basis required')
    f, r = f / np.linalg.norm(f), r / np.linalg.norm(r)
    if abs(f @ r) > .01:
        raise ValueError('Horizontal camera basis is not orthogonal')
    target, valid = np.zeros((len(distances), 2)), np.zeros(len(distances), bool)
    start = int(np.searchsorted(t, clock, side='right') - 1)
    if start < 0 or start >= len(t) - 1:
        return target, valid
    good = np.array([row.get('phase') == 'driving' and row.get('ai_controlled') is True
                     and row.get('wheels_outside') == 0 and row.get('paused') is False
                     for row in telemetry])
    delta = np.diff(t)
    step = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)
    edges = good[:-1] & good[1:] & (delta <= .1) & (step <= 60 * delta + .01)
    if not edges[start]:
        return target, valid
    end = start
    while end < len(edges) and edges[end]:
        end += 1
    fraction = (clock - t[start]) / delta[start]
    origin = p[start, [0, 2]] + fraction * (p[start + 1, [0, 2]] - p[start, [0, 2]])
    points = np.vstack([origin, p[start + 1:end + 1, [0, 2]]])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    for index, distance in enumerate(distances):
        if distance > arc[-1]:
            continue
        upper = int(np.searchsorted(arc, distance, side='left'))
        alpha = (distance - arc[upper - 1]) / (arc[upper] - arc[upper - 1])
        displacement = points[upper - 1] + alpha * (points[upper] - points[upper - 1]) - origin
        forward, lateral = displacement @ f, displacement @ r
        chord2 = float(displacement @ displacement)
        if forward > 0 and chord2 > .01:
            target[index] = np.arctan2(lateral, forward), 2 * lateral / chord2
            valid[index] = True
    return target, valid
