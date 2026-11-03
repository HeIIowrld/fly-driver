"""Camera-local hexagonal sampling with explicit body rays, not measured fly optics."""

import numpy as np

from .sensor_contract import readonly_array

VIEW_ORDER = ("front", "left", "right", "rear")


class FlyvisCameraProjector:
    """Preserve author's u/v receptor order and screen orientation per independent view.

    The lattice fills each camera's normalized image. Horizontal and vertical angular
    scale therefore depend on that camera. This is an engineering embodiment, not
    the author's 5.8-degree arena spacing or measured bilateral ommatidial axes.
    """

    def __init__(self, camera_config, u, v, *, extent=15):
        if type(extent) is not int or extent < 1:
            raise ValueError("Positive integral retinal extent required")
        expected = [
            (q, r)
            for q in range(-extent, extent + 1)
            for r in range(max(-extent, -extent - q), min(extent, extent - q) + 1)
        ]
        actual = np.column_stack([u, v])
        if not np.array_equal(actual, expected):
            raise ValueError("Author receptor order/coordinates differ from declared lattice")
        if camera_config["view_order"] != list(VIEW_ORDER):
            raise ValueError("Four independent source identities required")
        spacing = 1 / (2 * extent + 2)
        self.uv = readonly_array(
            np.column_stack(
                [0.5 + np.asarray(v) * spacing, 0.5 + (np.asarray(u) + np.asarray(v) / 2) * spacing]
            )
        )
        offsets, weights = [], []
        for x, wx in zip((-np.sqrt(3), 0, np.sqrt(3)), (1 / 6, 2 / 3, 1 / 6), strict=True):
            for y, wy in zip((-np.sqrt(3), 0, np.sqrt(3)), (1 / 6, 2 / 3, 1 / 6), strict=True):
                offsets.append([x * spacing * 0.35, y * spacing * 0.35])
                weights.append(wx * wy)
        self.samples = self.uv[:, None, :] + np.array(offsets)[None]
        if self.samples.min() <= 0 or self.samples.max() >= 1:
            raise ValueError("Acceptance support extends outside camera image")
        self.weights = np.array(weights)
        rays = []
        for name in VIEW_ORDER:
            camera = camera_config["views"][name]
            px, py = camera["projection_x"], camera["projection_y"]
            basis = np.array([camera["forward_body"], camera["right_body"], camera["up_body"]])
            if (
                not np.isfinite([px, py]).all()
                or min(px, py) <= 0
                or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(basis), -1.0, atol=1e-6)
            ):
                raise ValueError("Reviewed forward/right/up perspective camera required")
            ray = (
                basis[0]
                + (2 * self.uv[:, 0:1] - 1) / px * basis[1]
                + (1 - 2 * self.uv[:, 1:2]) / py * basis[2]
            )
            rays.append(ray / np.linalg.norm(ray, axis=1, keepdims=True))
        self.directions_body = readonly_array(rays)

    def sample(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (180, 320, 3):
            raise ValueError("Exact unmodified 320x180 RGB camera frame required")
        h, w = rgb.shape[:2]
        x = self.samples[:, :, 0] * w - 0.5
        y = self.samples[:, :, 1] * h - 0.5
        if x.min() < 0 or y.min() < 0 or x.max() > w - 1 or y.max() > h - 1:
            raise ValueError("Sampling support outside source pixels; no fill permitted")
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        wx, wy = (x - x0)[..., None], (y - y0)[..., None]
        pixels = (
            rgb[y0, x0] * (1 - wx) * (1 - wy)
            + rgb[y0, x1] * wx * (1 - wy)
            + rgb[y1, x0] * (1 - wx) * wy
            + rgb[y1, x1] * wx * wy
        )
        # Weighted LDR RGB intensity, not calibrated linear photon flux.
        intensity = pixels @ np.array([0.2126, 0.7152, 0.0722]) / 255.0
        return readonly_array(intensity @ self.weights, np.float32)


def integration_intervals(elapsed, maximum_step=0.005):
    """Internal model ticks preserve elapsed time; they are not new camera frames."""
    if not np.isfinite([elapsed, maximum_step]).all() or not 0 < elapsed <= 0.15:
        raise ValueError("Positive bounded original elapsed time required")
    if not 0 < maximum_step <= 0.005:
        raise ValueError("Author-network integration step must be at most 5ms")
    full = int(np.floor(elapsed / maximum_step))
    steps = [maximum_step] * full
    remainder = elapsed - full * maximum_step
    if remainder > 1e-12:
        steps.append(remainder)
    if not steps:
        steps.append(elapsed)
    return steps
