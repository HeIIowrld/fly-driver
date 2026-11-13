"""Intersect the installed surface triangles with body-horizontal cross-sections."""
from collections import defaultdict

import numpy as np

CELL = 20.
SEAM_EPS = .003


def coordinates(body):
    origin = np.asarray(body["renderer_only_position"], dtype=float)
    forward = np.asarray(body["renderer_only_look"], dtype=float)[[2, 0]]
    length = np.linalg.norm(forward)
    if not np.isfinite(origin).all() or not np.isfinite(length) or length < 1e-5:
        raise ValueError("Missing original local geometry reference")
    forward /= length
    left = np.array([-forward[1], forward[0]])
    return origin, forward, left


def intersect(triangles, station):
    """Return each nondegenerate section segment as (y lateral, z height) endpoints."""
    a, b = triangles, np.roll(triangles, -1, axis=1)
    keep = (a[:, :, 0] <= station) & (station < b[:, :, 0])
    keep |= (b[:, :, 0] <= station) & (station < a[:, :, 0])
    valid = keep.sum(axis=1) == 2
    if not valid.any():
        return np.empty((0, 2, 2)), valid
    start, stop = a[valid][keep[valid]], b[valid][keep[valid]]
    fraction = ((station - start[:, 0]) / (stop[:, 0] - start[:, 0]))[:, None]
    points = (start[:, 1:] + fraction * (stop[:, 1:] - start[:, 1:])).reshape(-1, 2, 2)
    points = np.take_along_axis(points, points[:, :, 0].argsort(axis=1)[:, :, None], axis=1)
    return points, valid


def merge(segments, classes):
    """Merge surface intervals while retaining the mesh class at each outer endpoint."""
    regions = []
    for i in np.argsort(segments[:, 0, 0]):
        start, stop = segments[i]
        surface = classes[i]
        if stop[0] - start[0] < 1e-7:
            continue
        if not regions or start[0] > regions[-1][1][0] + SEAM_EPS:
            regions.append((start, stop, surface, surface))
        elif stop[0] > regions[-1][1][0]:
            a, _, sa, _ = regions[-1]
            regions[-1] = (a, stop, sa, surface)
    return regions


class Surfaces:
    def __init__(self, meshes, configuration):
        tri, classes = [], []
        self.keys = tuple(configuration)
        self.valid = np.array([configuration[k]["valid"] for k in self.keys])
        for key, geometry in meshes:
            if key not in configuration:
                continue
            # Walls and edge-on triangles cannot supply a surface beneath a tyre.
            xz = geometry[:, :, [0, 2]]
            a, b = xz[:, 1] - xz[:, 0], xz[:, 2] - xz[:, 0]
            ground = abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]) > 1e-6
            tri.append(geometry[ground])
            classes.append(np.full(ground.sum(), self.keys.index(key), dtype=np.uint8))
        self.triangles = np.concatenate(tri)
        self.classes = np.concatenate(classes)
        if not np.isfinite(self.triangles).all():
            raise ValueError("Nonfinite KN5 surface coordinate")
        low = np.floor(self.triangles[:, :, [0, 2]].min(axis=1) / CELL).astype(int)
        high = np.floor(self.triangles[:, :, [0, 2]].max(axis=1) / CELL).astype(int)
        self.cells = defaultdict(list)
        for i, ((x0, z0), (x1, z1)) in enumerate(zip(low, high, strict=True)):
            for x in range(x0, x1+1):
                for z in range(z0, z1+1):
                    self.cells[x, z].append(i)

    def sections(self, body, stations):
        origin, forward, left = coordinates(body)
        footprint = np.array([origin[[2, 0]] + forward*d + left*y
                              for d in (min(stations), max(stations)) for y in (-24., 24.)])[:, ::-1]
        low = np.floor(footprint.min(axis=0) / CELL).astype(int)
        high = np.floor(footprint.max(axis=0) / CELL).astype(int)
        indices = sorted({i for x in range(low[0], high[0]+1) for z in range(low[1], high[1]+1)
                          for i in self.cells.get((x, z), ())})
        offsets = self.triangles[indices] - origin
        classes = self.classes[indices]
        xy = offsets[:, :, [2, 0]]
        local = np.stack((xy @ forward, xy @ left, offsets[:, :, 1]), axis=-1)
        low_x, high_x = local[:, :, 0].min(axis=1), local[:, :, 0].max(axis=1)
        nearby = (local[:, :, 1].max(axis=1) >= -24.) & (local[:, :, 1].min(axis=1) <= 24.)
        nearby &= (local[:, :, 2].max(axis=1) >= -4.) & (local[:, :, 2].min(axis=1) <= 4.)
        center = 0.
        results = []
        for station in stations:
            selected = nearby & (low_x <= station) & (high_x > station)
            segments, usable = intersect(local[selected], station)
            kinds = classes[selected][usable]
            valid = self.valid[kinds]
            regions = merge(segments[valid], kinds[valid])
            match = [r for r in regions if r[0][0]-SEAM_EPS <= center <= r[1][0]+SEAM_EPS]
            if len(match) != 1:
                results.append(None)
                continue
            first, last, right_kind, left_kind = match[0]
            if not 2. <= last[0] - first[0] <= 24.:
                results.append(None)
                continue
            center = (first[0]+last[0]) / 2.
            sides = []
            for side, (point, kind) in enumerate(((first, right_kind), (last, left_kind))):
                edge, inward = point[0], 1 if side == 0 else -1
                outside = segments[~valid]
                outside_class = kinds[~valid]
                if len(outside):
                    extends = (outside[:, 0, 0] < edge - .025) if side == 0 else (outside[:, 1, 0] > edge + .025)
                    end = outside[:, 1-side]
                    distance = np.abs(end[:, 0] - edge)
                    eligible = extends & (distance <= .05) & (abs(end[:, 1] - point[1]) <= .25)
                    names = sorted(set(self.keys[k] for k in outside_class[eligible]))
                else:
                    names = []
                world = origin.copy()
                world[[2, 0]] += station * forward + edge * left
                world[1] += point[1]
                sides.append(dict(lateral_m=float(edge), world=world.tolist(), inside=self.keys[kind],
                                  outside=names, surface_transition_confirmed=len(names) == 1))
            results.append(sides)
        return results
