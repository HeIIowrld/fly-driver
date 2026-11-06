"""Privileged AC Gym teacher geometry; never a fly actor observation adapter.

Gap priorities and ray arithmetic follow the MIT-licensed legacy AC Gym recipe
a90db0a8b164164e11412857b396bf6874462976. Retained source/license live under
assets/external/ac-teacher-geometry-20261027. No upstream module is imported.
"""

import numpy as np

from .graph import sha256


def legacy_signed_gap(point, line, *, cyclic=False):
    """Preserve nearest-pivot/adjacent-segment priority and legacy sign convention."""
    point, line = np.asarray(point, float), np.asarray(line, float)
    if (point.shape != (2,) or line.ndim != 2 or line.shape[1] != 2 or len(line) < 3
            or not np.isfinite(point).all() or not np.isfinite(line).all()):
        raise ValueError('Finite teacher point and reference line required')
    pivot = int(np.argmin(np.linalg.norm(line - point, axis=1)))
    if not cyclic:
        pivot = min(pivot, len(line) - 2)

    def segment(start, end):
        vector, relative = end - start, point - start
        length = np.linalg.norm(vector)
        if length <= 0:
            raise ValueError('Legacy reference has a zero-length adjacent segment')
        fraction = np.clip(np.sum((vector / length) * (relative / length)), 0, 1)
        nearest = start + vector * fraction
        sign = np.sign(vector[0] * relative[1] - vector[1] * relative[0])
        return nearest, fraction, sign, np.linalg.norm(nearest - point)

    earlier = segment(line[pivot - 1], line[pivot])
    later = segment(line[pivot], line[(pivot + 1) % len(line)])
    earlier_inside, later_inside = (0 < item[1] < 1 for item in (earlier, later))
    chosen = earlier if earlier_inside else later
    if earlier_inside and later_inside:
        chosen = earlier if earlier[3] <= later[3] else later
    return float(chosen[3] * earlier[2] * -1)


class PrivilegedTrackGeometry:
    def __init__(self, track_csv, reference_csv, *, track_sha256, reference_sha256, cyclic_reference=False):
        if sha256(track_csv) != track_sha256 or sha256(reference_csv) != reference_sha256:
            raise ValueError('Teacher track/reference hashes differ')
        track = np.genfromtxt(track_csv, delimiter=',', names=True)
        reference = np.genfromtxt(reference_csv, delimiter=',', names=True)
        self.line = np.column_stack([reference['pos_x'], reference['pos_y']])
        self.cyclic_reference = cyclic_reference
        if cyclic_reference:
            if len(self.line)<4 or not np.array_equal(self.line[0],self.line[-1]):
                raise ValueError('Explicit duplicate closure required for cyclic teacher geometry')
            self.line = self.line[:-1]
            if np.any(np.linalg.norm(np.roll(self.line,-1,axis=0)-self.line,axis=1)<=0):
                raise ValueError('Cyclic reference has a zero-length segment')
        columns = []
        for name in ('right_border_x', 'right_border_y', 'left_border_x', 'left_border_y'):
            full = np.asarray(track[name])
            if not 3 <= len(full) <= 100000 or not np.isfinite(full).all():
                raise ValueError('Bounded finite teacher border required')
            selected = full[::10]
            if selected[-1] != full[-1]:
                selected = np.r_[selected, full[-1]]
            columns.append(selected)
        if len({len(x) for x in columns}) != 1 or not np.isfinite(self.line).all():
            raise ValueError('Teacher geometry dimensions differ')
        borders = [np.column_stack(columns[start:start + 2]) for start in (0, 2)]
        self.walls = np.concatenate([
            np.column_stack([border[:-1], border[1:]]) for border in borders])

    def features(self, x, y, yaw):
        if not np.isfinite([x, y, yaw]).all():
            raise ValueError('Finite privileged pose required')
        angles = np.linspace(0, np.pi, 11) + yaw - np.pi / 2
        cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
        x1, y1, x2, y2 = self.walls.T
        dx, dy = x2 - x1, y2 - y1
        determinant = cosine * dy - sine * dx
        determinant[determinant == 0] = 1e-8
        rx, ry = x1 - x, y1 - y
        distance = (rx * dy - ry * dx) / determinant
        fraction = (rx * sine - ry * cosine) / determinant
        valid = (fraction >= 0) & (fraction <= 1) & (distance >= 0)
        distance[~valid] = 200.
        nearest = distance.min(1)
        # Reproduce the upstream intersection-to-origin norm, including rounding.
        intersection = np.stack([x + cosine[:, 0] * nearest, y + sine[:, 0] * nearest], -1)
        sensors = np.linalg.norm(intersection - [x, y], axis=1)
        return dict(gap=legacy_signed_gap([x, y], self.line, cyclic=self.cyclic_reference), sensors=sensors)
