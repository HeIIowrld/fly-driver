"""Privileged offline diagnostic only: original future reference points onto original pixels."""

import numpy as np

from .ac_csp_geometry import matrix


class FutureTrace:
    def __init__(self, telemetry):
        self.times = np.asarray([r['script_clock'] for r in telemetry], float)
        self.positions = np.asarray([r['position'] for r in telemetry], float)
        if (len(self.times) < 2 or self.positions.shape != (len(self.times), 3)
                or not np.isfinite(self.times).all() or not np.isfinite(self.positions).all()
                or not (np.diff(self.times) > 0).all()):
            raise ValueError('Original finite ordered three-dimensional telemetry required')
        good = np.array([r.get('phase') == 'driving' and r.get('ai_controlled') is True
                         and r.get('paused') is False and r.get('wheels_outside') == 0
                         for r in telemetry])
        dt = np.diff(self.times)
        horizontal_steps = np.linalg.norm(np.diff(self.positions[:, [0, 2]], axis=0), axis=1)
        self.edges = good[:-1] & good[1:] & (dt <= .1) & (horizontal_steps <= 60*dt+.01)

    def at(self, clock, distances=(5., 10., 20.)):
        distances = np.asarray(distances, float)
        if (not np.isfinite(clock) or distances.ndim != 1
                or not np.isfinite(distances).all() or not (distances > 0).all()):
            raise ValueError('Finite original camera clock and positive distances required')
        result, valid = np.zeros((len(distances), 3)), np.zeros(len(distances), bool)
        index = int(np.searchsorted(self.times, clock, side='right')-1)
        if index < 0 or index >= len(self.edges) or not self.edges[index]:
            return np.zeros(3), result, valid
        fraction = (clock-self.times[index])/(self.times[index+1]-self.times[index])
        start = self.positions[index]*(1-fraction)+self.positions[index+1]*fraction
        previous, arc = start, 0.
        for edge in range(index, len(self.edges)):
            if not self.edges[edge]:
                break
            point = self.positions[edge+1]
            length = np.linalg.norm((point-previous)[[0, 2]])
            if length > 0:
                for h, distance in enumerate(distances):
                    if not valid[h] and arc < distance <= arc+length:
                        result[h] = previous+(distance-arc)/length*(point-previous)
                        valid[h] = True
            arc += length
            previous = point
            if arc >= distances.max():
                break
        return start, result, valid


def project_reference_points(points, view):
    """CSP row-vector projection; preserves recorded origin, no fitted pixel offset."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite original world reference positions required')
    before = view.get('renderer_origin_shift_before_diagnostic')
    after = view.get('renderer_origin_shift_after_diagnostic')
    if before is None and after is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(before, float)
        if (origin.shape != (3,) or not np.isfinite(origin).all()
                or not np.array_equal(origin, np.asarray(after, float))):
            raise ValueError('Stable explicitly recorded render origin required')
    homogeneous = np.column_stack([points+origin, np.ones(len(points))])
    clip = homogeneous @ matrix(view['view_matrix_diagnostic']) @ matrix(
        view['projection_matrix_diagnostic'])
    nonzero = np.abs(clip[:, 3]) > 1e-12
    ndc = np.divide(clip[:, :2], clip[:, 3, None],
                    out=np.zeros((len(points), 2)), where=nonzero[:, None])
    uv = np.column_stack([.5+.5*ndc[:, 0], .5-.5*ndc[:, 1]])
    in_front = clip[:, 3] > 1e-12
    in_image = in_front & (uv >= 0).all(1) & (uv <= 1).all(1)
    return uv, in_front, in_image
