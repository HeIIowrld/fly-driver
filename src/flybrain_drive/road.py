"""Synthetic heightfield road. Metres, world X forward, Y left, Z up."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RoadConfig:
    length: float = 100.0
    width: float = 6.0
    grade: float = 0.0  # ratio, not degrees
    hill_amplitude: float = 0.6
    hill_period: float = 70.0
    bank_degrees: float = 6.0
    turn_amplitude: float = 2.0
    turn_period: float = 70.0
    bump_height: float = 0.03
    flattened: bool = False

    def __post_init__(self):
        if self.length <= 10 or self.width <= 2 or min(self.hill_period, self.turn_period) <= 0:
            raise ValueError("Invalid road dimensions/periods")
        if abs(self.grade) > 0.2 or abs(self.bank_degrees) > 15:
            raise ValueError("Prototype is limited to grade <=20%, bank <=15 degrees")


class SyntheticRoad:
    provenance = "synthetic_v1_not_a_measured_circuit"

    def __init__(self, config=None):
        self.config = config or RoadConfig()
        self.x = np.linspace(0, self.config.length, 2001)
        y = self.center_y(self.x)
        self.xyz = np.column_stack((self.x, y, self.height(self.x, y)))
        delta = np.diff(self.xyz, axis=0)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(delta, axis=1))]
        self._segments = delta
        self._segment_squared_lengths = np.sum(delta * delta, axis=1)

    def center_y(self, x):
        c = self.config
        return c.turn_amplitude * np.sin(2 * np.pi * np.asarray(x) / c.turn_period)

    def height(self, x, y):
        c = self.config
        x, y = np.asarray(x), np.asarray(y)
        if c.flattened:
            return np.zeros(np.broadcast_shapes(x.shape, y.shape))
        bank = np.deg2rad(c.bank_degrees) * np.sin(2 * np.pi * x / c.length)
        bump = c.bump_height * np.exp(-(((x - 50) / 0.5) ** 2) - ((y - 1) / 0.4) ** 2)
        return (
            c.grade * x
            + c.hill_amplitude * np.sin(2 * np.pi * x / c.hill_period)
            + (y - self.center_y(x)) * np.tan(bank)
            + bump
        )

    def frame(self, x):
        """Columns: road tangent, left, up. All are world vectors."""
        y = float(self.center_y(x))
        eps = 0.01
        hx = float((self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps))
        hy = float((self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps))
        normal = np.array([-hx, -hy, 1.0])
        normal /= np.linalg.norm(normal)
        dy = float((self.center_y(x + eps) - self.center_y(x - eps)) / (2 * eps))
        tangent = np.array([1.0, dy, hx + hy * dy])
        tangent /= np.linalg.norm(tangent)
        left = np.cross(normal, tangent)
        left /= np.linalg.norm(left)
        return np.column_stack((tangent, left, np.cross(tangent, left)))

    def project(self, position):
        """3D nearest polyline projection; no XY-only progress."""
        p = np.asarray(position, dtype=float)
        segments = self._segments
        t = np.clip(
            np.sum((p - self.xyz[:-1]) * segments, axis=1) / self._segment_squared_lengths,
            0,
            1,
        )
        candidates = self.xyz[:-1] + t[:, None] * segments
        i = int(np.argmin(np.sum((p - candidates) ** 2, axis=1)))
        point = candidates[i]
        frame = self.frame(float(point[0]))
        progress = float(self.s[i] + t[i] * (self.s[i + 1] - self.s[i]))
        lateral = float(np.dot(p - point, frame[:, 1]))
        return progress, lateral, point, frame

    def export(self):
        frames = np.stack([self.frame(float(x)) for x in self.x])
        return {
            "provenance": self.provenance,
            "config": asdict(self.config),
            "centerline_xyz": self.xyz.tolist(),
            "progress_3d": self.s.tolist(),
            "left_boundary_xyz": (self.xyz + frames[:, :, 1] * self.config.width / 2).tolist(),
            "right_boundary_xyz": (self.xyz - frames[:, :, 1] * self.config.width / 2).tolist(),
            "normals": frames[:, :, 2].tolist(),
        }
