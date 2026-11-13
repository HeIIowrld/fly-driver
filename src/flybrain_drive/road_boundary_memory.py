"""Short-lived observed road sections in a relative odometry frame."""

from collections import deque

import numpy as np


def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class RoadBoundaryMemory:
    """Fill missing sections only by interpolation between observed edge points."""

    def __init__(self, max_age_s):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Positive finite boundary memory age required")
        self.max_age_s = max_age_s
        self.reset()

    def reset(self):
        self.pose = np.zeros(3)
        self.time_s = 0.
        self.frames = deque()

    def advance(self, speed_mps, yaw_left_rad_s, elapsed_s):
        if (not np.isfinite([speed_mps, yaw_left_rad_s, elapsed_s]).all()
                or speed_mps < 0 or not 0 < elapsed_s <= .08):
            raise ValueError("Fresh finite speed, yaw and interval up to 80ms required")
        turn = yaw_left_rad_s * elapsed_s
        angle = self.pose[2] + .5*turn
        self.pose[:2] += speed_mps*elapsed_s*np.array([np.cos(angle), np.sin(angle)])
        self.pose[2] += turn
        self.time_s += elapsed_s
        while self.frames and self.time_s - self.frames[0][0] > self.max_age_s:
            self.frames.popleft()

    def update(self, sections, *, near_conflict_m=None, near_conflict_max_age_s=.24):
        """Optionally hold a fresh carried nearest section on a large disagreement."""
        if near_conflict_m is not None and (
            not np.isfinite([near_conflict_m, near_conflict_max_age_s]).all()
            or min(near_conflict_m, near_conflict_max_age_s) <= 0
        ):
            raise ValueError("Positive finite near-section conflict limits required")
        observed = []
        previous = -np.inf
        for section in sections:
            x = section["forward_x_m"]
            if not np.isfinite(x) or x <= max(0., previous):
                raise ValueError("Positive ordered section distances required")
            previous = x
            if section["valid"]:
                right, left = section["right_y_m"], section["left_y_m"]
                if not np.isfinite([right, left]).all() or right >= left:
                    raise ValueError("Finite ordered observed boundaries required")
                observed.append(np.array([[x, right], [x, left]]))
            else:
                observed.append(None)
        rotation = _rotation(self.pose[2])
        recovered = []
        for index, section in enumerate(sections):
            x = section["forward_x_m"]
            direct = section["valid"]
            check_conflict = direct and index == 0 and near_conflict_m is not None
            if direct and not check_conflict:
                recovered.append(dict(section, source="observed", age_s=0.))
                continue
            fill = None
            for time_s, segments in reversed(self.frames):
                if check_conflict and self.time_s - time_s > near_conflict_max_age_s:
                    break
                local = (segments - self.pose[:2]) @ rotation
                start, end = local[:, 0], local[:, 1]
                lo, hi = start[:, :, 0], end[:, :, 0]
                hits = (lo <= x) & (x < hi)
                if np.all(hits.sum(axis=0) == 1):
                    row, edge = hits.argmax(axis=0), np.arange(2)
                    fraction = (x-lo[row, edge])/(hi[row, edge]-lo[row, edge])
                    right, left = start[row, edge, 1]+fraction*(end[row, edge, 1]-start[row, edge, 1])
                    if right >= left:
                        continue
                    fill = dict(forward_x_m=x, right_y_m=float(right), left_y_m=float(left),
                                valid=True, source="carried", age_s=self.time_s-time_s)
                    break
            if direct:
                if fill is not None and max(abs(fill[k] - section[k])
                                            for k in ("right_y_m", "left_y_m")) > near_conflict_m:
                    observed[index] = None
                    recovered.append(dict(fill, direct_conflict=True))
                else:
                    recovered.append(dict(section, source="observed", age_s=0.))
            else:
                recovered.append(fill if fill is not None else dict(section, source="missing", age_s=None))
        segments = [[a, b] for a, b in zip(observed, observed[1:]) if a is not None and b is not None]
        if segments:
            self.frames.append((self.time_s, np.asarray(segments) @ rotation.T + self.pose[:2]))
        return recovered
