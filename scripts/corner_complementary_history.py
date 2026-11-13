"""Causal independent-side fusion of vehicle-relative road boundaries."""

from collections import deque

import numpy as np

from corner_fusion_boundaries import advance, stations


def _resample(points):
    result = np.full((len(stations), 2), np.nan)
    for side in range(2):
        samples = points[:, side]
        samples = samples[np.isfinite(samples).all(axis=1)]
        if not len(samples):
            continue
        x, y = samples[np.argsort(samples[:, 0])].T
        for i, query in enumerate(stations):
            j = np.searchsorted(x, query)
            if j < len(x) and abs(x[j]-query) < 1e-8:
                result[i, side] = y[j]
            elif 0 < j < len(x) and 0. < x[j]-x[j-1] <= 4.1:
                result[i, side] = y[j-1] + (query-x[j-1])*(y[j]-y[j-1])/(x[j]-x[j-1])
    return result


class EdgeHistory:
    def __init__(self, max_age=.24):
        self.max_age = max_age
        self.frames = deque()

    def update(self, timestamp, current, step):
        """Advance past observations, then fill missing sides without recycling estimates."""
        if step is None:
            self.frames.clear()
        else:
            rotation, move = step
            self.frames = deque((t, advance(p, rotation, move)) for t,p in self.frames
                                if 0 < timestamp-t <= self.max_age)
        counts = np.zeros(current.shape, dtype=int)
        proposals = {k:current.copy() for k in (1, 2)}
        if self.frames:
            history = np.stack([_resample(p) for _,p in self.frames])
            for i, side in zip(*np.where(~np.isfinite(current)), strict=True):
                observed = history[:,i,side]
                observed = observed[np.isfinite(observed)]
                counts[i,side] = len(observed)
                if len(observed) and np.ptp(observed) <= .8:
                    estimate = float(np.median(observed))
                    for minimum, edges in proposals.items():
                        if len(observed) >= minimum:
                            edges[i,side] = estimate
        for edges in proposals.values():
            width = edges[:,1]-edges[:,0]
            invalid = np.isfinite(width) & ((width <= 3.) | (width >= 20.))
            edges[invalid] = current[invalid]
        points = np.stack((np.broadcast_to(stations[:,None],current.shape),current),axis=-1)
        self.frames.append((timestamp,points))
        return proposals, counts
