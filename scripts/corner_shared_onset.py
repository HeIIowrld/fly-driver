"""Refit distance for a known bend direction using a shared piecewise road curve."""

import numpy as np


def refine_distance(stations, edges, direction, distance):
    """Require a common onset to fit better than a steady bend on both observed edges."""
    prepared = []
    for side in range(2):
        valid = np.isfinite(edges[:,side])
        x = stations[valid]
        if len(x) < 5 or np.max(np.diff(x)) > 4.:
            return None
        basis = np.column_stack((np.ones(len(x)), x))
        projector = np.eye(len(x)) - basis @ np.linalg.pinv(basis)
        prepared.append((x, projector @ edges[valid,side], projector))
    target = np.concatenate([y for _,y,_ in prepared])

    def fit(onset):
        hinge = np.concatenate([p @ (x*x if onset is None else np.maximum(x-onset,0.)**2)
                                for x,_,p in prepared])
        norm = hinge @ hinge
        if norm <= 1e-9:
            return None
        coefficient = (hinge @ target) / norm
        return float(np.sqrt(np.mean((target-coefficient*hinge)**2))),float(coefficient)

    null = fit(None)
    if null is None:
        return None
    best = None
    sign = 1 if direction == 1 else -1
    for onset in np.arange(4.,21.):
        if abs(onset-distance)>6.:
            continue
        if not all((x<=onset).sum()>=2 and (x>onset).sum()>=3
                   and onset-x[x<=onset][-1]<=2. and x[x>onset][0]-onset<=2.
                   for x,_,_ in prepared):
            continue
        match = fit(onset)
        if match is None:
            continue
        error, coefficient = match
        if coefficient*sign<.001 or error>.75 or error>.8*null[0]:
            continue
        rank=(error,abs(onset-distance),float(onset))
        if best is None or rank<best:
            best=rank
    return best[-1] if best is not None else None
