"""Configurable ray sampling and synthetic motion filters; not measured fly optics."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    yaw_degrees: float = 25.0
    pitch_degrees: float = 0.0  # negative points toward the road
    fovy: float = 80.0
    baseline: float = 0.64
    forward: float = 0.4
    height: float = 0.2
    source_size: int = 0  # zero uses actor image size
    sampling: str = "perspective"
    angular_width: float = 110.0
    angular_height: float = 90.0
    near_m: float = 0.0  # both zero preserve historical extent-scaled renderer defaults
    far_m: float = 0.0

    def __post_init__(self):
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Camera parameters must be finite")
        if self.sampling not in ("perspective", "angular_rect", "angular_hex"):
            raise ValueError("Unknown camera sampling")
        if not 1 <= self.fovy < 170 or self.baseline < 0 or self.source_size < 0:
            raise ValueError("Invalid camera geometry")
        if self.source_size and (
            self.source_size < 16 or int(self.source_size) != self.source_size
        ):
            raise ValueError("Camera source_size must be an integer >=16")
        if not 0 < self.angular_width < 170 or not 0 < self.angular_height < 170:
            raise ValueError("Invalid angular sampling span")
        if not ((self.near_m == self.far_m == 0) or 0 < self.near_m < self.far_m):
            raise ValueError("Specify positive near_m < far_m, or both zero")


class RaySampler:
    def __init__(self, camera, size):
        self.camera, self.size = camera, size
        self.source_size = camera.source_size or size
        row, col = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        tangent = np.tan(np.deg2rad(camera.fovy / 2))
        if camera.sampling.startswith("angular_"):
            # Staggered rows approximate a hexagonal lattice in angular coordinates.
            offset = ((row % 2) - 0.5) * 0.5 if camera.sampling == "angular_hex" else 0
            azimuth = np.deg2rad(
                ((col + 0.5 + offset) / size - 0.5) * camera.angular_width
            )
            elevation = np.deg2rad((0.5 - (row + 0.5) / size) * camera.angular_height)
            rays = np.stack(
                (
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                    -np.cos(elevation) * np.cos(azimuth),
                ),
                axis=-1,
            )
            self.x = (0.5 + rays[..., 0] / -rays[..., 2] / tangent / 2) * self.source_size - 0.5
            self.y = (0.5 - rays[..., 1] / -rays[..., 2] / tangent / 2) * self.source_size - 0.5
        else:
            self.x = (col + 0.5) / size * self.source_size - 0.5
            self.y = (row + 0.5) / size * self.source_size - 0.5
            rays = np.stack(
                ((2 * (col + 0.5) / size - 1) * tangent,
                 (1 - 2 * (row + 0.5) / size) * tangent, -np.ones_like(col)),
                axis=-1,
            )
        self.rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)
        self.valid = (
            (self.x >= 0) & (self.x <= self.source_size - 1)
            & (self.y >= 0) & (self.y <= self.source_size - 1)
        ).astype(np.float32)
        self.x = np.clip(self.x, 0, self.source_size - 1)
        self.y = np.clip(self.y, 0, self.source_size - 1)
        self.x0, self.y0 = np.floor(self.x).astype(int), np.floor(self.y).astype(int)
        self.x1 = np.minimum(self.x0 + 1, self.source_size - 1)
        self.y1 = np.minimum(self.y0 + 1, self.source_size - 1)
        self.wx, self.wy = self.x - self.x0, self.y - self.y0

    def sample(self, source):
        if source.shape != (self.source_size, self.source_size, 3):
            raise ValueError("Unexpected camera source dimensions")
        if self.camera.sampling == "perspective" and self.source_size == self.size:
            return source.copy()
        wx, wy = self.wx[..., None], self.wy[..., None]
        pixels = (
            source[self.y0, self.x0] * (1 - wx) * (1 - wy)
            + source[self.y0, self.x1] * wx * (1 - wy)
            + source[self.y1, self.x0] * (1 - wx) * wy
            + source[self.y1, self.x1] * wx * wy
        )
        return (np.rint(pixels) * self.valid[..., None]).astype(np.uint8)


def neighbor(values, dx, dy):
    """Zero boundary, no circular wrap. Last two dimensions are spatial."""
    result = np.zeros_like(values)
    height, width = values.shape[-2:]
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    result[..., y0:y1, x0:x1] = values[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


class MotionFilters:
    channels = ["on", "off", "right", "left", "up", "down"]

    def __init__(self, spatial_valid, staggered=False, adaptation_tau=0.15, delay_tau=0.04):
        self.valid = np.asarray(spatial_valid, np.float32)
        self.staggered = staggered
        self.adaptation_tau, self.delay_tau = adaptation_tau, delay_tau
        self.reset()

    def reset(self):
        self.last_time = None
        self.adaptation = None
        self.delayed = None

    def adjacent(self, values, dx, dy):
        first = neighbor(values, dx, dy)
        if not self.staggered or dy == 0:
            return first
        parity = np.arange(values.shape[-2]) % 2
        extra = np.where(parity[:, None] == 1, neighbor(values, 1, dy), neighbor(values, -1, dy))
        return (first + extra) / 2

    def update(self, images, timestamp):
        luminance = np.mean(images.astype(np.float32), axis=-1) / 255 * self.valid
        features = np.zeros((*luminance.shape, 6), np.float32)
        feature_valid = np.zeros_like(features)
        if self.last_time is None:
            self.adaptation = luminance.copy()
            self.delayed = np.zeros((2, *luminance.shape), np.float32)
        else:
            dt = timestamp - self.last_time
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Motion filters need increasing exposure timestamps")
            contrast = luminance - self.adaptation
            current = np.stack((np.maximum(contrast, 0), np.maximum(-contrast, 0)))
            features[..., :2] = np.moveaxis(current, 0, -1)
            feature_valid[..., :2] = self.valid[..., None]
            for index, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, -1), (0, 1)), 2):
                supported = self.valid * (self.adjacent(self.valid, dx, dy) >= 1 - 1e-6)
                correlation = np.sum(
                    self.delayed * self.adjacent(current, dx, dy)
                    - current * self.adjacent(self.delayed, dx, dy), axis=0
                )
                features[..., index] = np.clip(16 * correlation, 0, 1) * supported
                feature_valid[..., index] = supported
            self.adaptation += (1 - np.exp(-dt / self.adaptation_tau)) * contrast
            self.delayed += (1 - np.exp(-dt / self.delay_tau)) * (current - self.delayed)
        self.last_time = float(timestamp)
        return features, feature_valid
