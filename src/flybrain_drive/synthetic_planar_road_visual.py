"""Controlled four-view road images; synthetic and not a measured circuit."""

import math

import numpy as np

from .flyvis_camera import VIEW_ORDER


class SyntheticPlanarRoadVisual:
    """Flat ground and curved road edges viewed through declared camera rays.

    Coordinates are metres in an engineering X-forward, Y-left, Z-up frame.
    The generated bend is diagnostic truth and is never part of an actor input.
    It models neither vehicle dynamics nor a scanned Assetto Corsa circuit.
    """

    def __init__(self, camera, *, width_m=8.0, bend_start_m=12.0, curvature=0.018):
        if (camera.get('view_order') != list(VIEW_ORDER)
                or not all(math.isfinite(v) for v in (width_m, bend_start_m, curvature))
                or not 3 <= width_m <= 20 or not 3 <= bend_start_m <= 30
                or not 0 < curvature <= .05):
            raise ValueError('Four declared views and bounded synthetic road required')
        self.width_m, self.bend_start_m, self.curvature = width_m, bend_start_m, curvature
        y, x = np.mgrid[:180, :320]
        self.rays, self.origins = [], []
        for name in VIEW_ORDER:
            view = camera['views'][name]
            basis = np.asarray([view['forward_body'], view['right_body'], view['up_body']], float)
            origin = np.asarray(view['origin_body_m'], float)
            px, py = view['projection_x'], view['projection_y']
            if (basis.shape != (3, 3) or origin.shape != (3,) or not np.isfinite(basis).all()
                    or not np.isfinite(origin).all() or origin[2] <= 0
                    or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)
                    or not np.isfinite([px, py]).all() or min(px, py) <= 0):
                raise ValueError('Finite orthonormal camera and positive height required')
            ray = (basis[0] + (2*(x[..., None]+.5)/320-1)/px*basis[1]
                   + (1-2*(y[..., None]+.5)/180)/py*basis[2])
            self.rays.append(ray / np.linalg.norm(ray, axis=-1, keepdims=True))
            self.origins.append(origin)

    def render(self, *, bend, forward_m, texture_seed, yaw_rad=0.0, lateral_m=0.0):
        if (bend not in (-1, 0, 1) or isinstance(bend, bool)
                or not math.isfinite(forward_m) or not 0 <= forward_m <= 20
                or isinstance(lateral_m, bool) or not isinstance(lateral_m, (int, float))
                or not math.isfinite(lateral_m) or abs(lateral_m) > 20
                or type(texture_seed) is not int or texture_seed < 0
                or isinstance(yaw_rad, bool) or not isinstance(yaw_rad, (int, float))
                or not math.isfinite(yaw_rad) or abs(yaw_rad) > math.pi):
            raise ValueError('Explicit bend, bounded displacement and texture seed required')
        random = np.random.default_rng(texture_seed)
        phase = random.uniform(-np.pi, np.pi, 2)
        colors = random.uniform(-8, 8, 3)
        frames = []
        if yaw_rad == 0:
            rotation = None
        else:
            c, s = math.cos(yaw_rad), math.sin(yaw_rad)
            rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        for ray, base in zip(self.rays, self.origins, strict=True):
            if rotation is not None:
                ray = ray @ rotation.T
                base = rotation @ base
            origin = base + np.array([forward_m, lateral_m, 0.])
            down = ray[..., 2] < -1e-6
            distance = np.divide(-origin[2], ray[..., 2], out=np.zeros_like(ray[..., 2]), where=down)
            ground = down & (distance <= 100.)
            x = origin[0] + distance*ray[..., 0]
            y = origin[1] + distance*ray[..., 1]
            center = bend*self.curvature*np.maximum(x-self.bend_start_m, 0.)**2
            offset = np.abs(y-center)
            road = ground & (offset <= self.width_m/2)
            edge = ground & (np.abs(offset-self.width_m/2) < .12)
            texture = 5*np.sin(.16*x+phase[0])*np.cos(.23*y+phase[1])
            rgb = np.empty((180, 320, 3), dtype=np.float64)
            rgb[:] = np.array([125., 160., 190.])+colors
            rgb[ground] = (np.array([62., 92., 49.])+colors)[None] + texture[ground, None]
            rgb[road] = (np.array([69., 69., 67.])+colors)[None] + texture[road, None]
            rgb[edge] = np.array([210., 210., 202.])+colors
            frames.append(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))
        return np.stack(frames)
