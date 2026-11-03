"""Direction-preserving compound-eye projection with finite angular acceptance.

Three-point Gaussian quadrature in each tangent direction approximates acceptance;
this is an engineering optical model, not measured Drosophila anatomy.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .sensor_contract import EyeSample, readonly_array, readonly_mask


@dataclass(frozen=True)
class EyeGeometry:
    side: str
    directions_body: np.ndarray
    acceptance_sigma_degrees: np.ndarray
    origin_body_m: np.ndarray
    provenance: str

    def __post_init__(self):
        directions = readonly_array(self.directions_body)
        sigma = readonly_array(self.acceptance_sigma_degrees)
        origin = readonly_array(self.origin_body_m)
        if self.side not in ("left", "right") or not self.provenance.strip():
            raise ValueError("Eye side and geometry provenance are required")
        if directions.ndim != 2 or directions.shape[1] != 3 or len(directions) < 1:
            raise ValueError("Eye directions must be N x 3")
        if not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-7):
            raise ValueError("Eye optical axes must be unit vectors")
        if sigma.shape != (len(directions),) or np.any((sigma <= 0) | (sigma > 15)):
            raise ValueError("Acceptance sigma must be per ray, positive and <=15 degrees")
        if origin.shape != (3,):
            raise ValueError("Eye origin must be a body-frame position in metres")
        object.__setattr__(self, "directions_body", directions)
        object.__setattr__(self, "acceptance_sigma_degrees", sigma)
        object.__setattr__(self, "origin_body_m", origin)

    def manifest(self):
        return {
            "side": self.side, "directions_body": self.directions_body.tolist(),
            "acceptance_sigma_degrees": self.acceptance_sigma_degrees.tolist(),
            "origin_body_m": self.origin_body_m.tolist(), "provenance": self.provenance,
            "body_axes": "X forward, Y left, Z up",
            "acceptance": "Gaussian sigma; 3x3 tangent-plane quadrature approximation",
        }

    @property
    def sha256(self):
        data = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PerspectiveView:
    camera_to_body: np.ndarray
    origin_body_m: np.ndarray
    fovy_degrees: float

    def __post_init__(self):
        rotation, origin = readonly_array(self.camera_to_body), readonly_array(self.origin_body_m)
        if (rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3))
                or not np.isclose(np.linalg.det(rotation), 1)):
            raise ValueError("Camera pose must be a proper rotation, with X forward/Y left/Z up")
        if origin.shape != (3,) or not np.isfinite(self.fovy_degrees):
            raise ValueError("Invalid perspective source calibration")
        if not 1 < self.fovy_degrees < 170:
            raise ValueError("Perspective vertical FOV outside supported range")
        object.__setattr__(self, "camera_to_body", rotation)
        object.__setattr__(self, "origin_body_m", origin)


class CompoundEyeProjector:
    def __init__(self, geometry, view, min_coverage=1 - 1e-9):
        if not np.allclose(geometry.origin_body_m, view.origin_body_m, atol=1e-9, rtol=0):
            raise ValueError("Translation between viewpoints requires depth; cannot invent stereo")
        if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError("Coverage threshold must be in (0, 1]")
        self.geometry, self.view, self.min_coverage = geometry, view, min_coverage
        directions = geometry.directions_body
        axis = np.tile([0., 0., 1.], (len(directions), 1))
        axis[np.abs(directions[:, 2]) > .9] = [0, 1, 0]
        tangent_a = np.cross(axis, directions)
        tangent_a /= np.linalg.norm(tangent_a, axis=1, keepdims=True)
        tangent_b = np.cross(directions, tangent_a)
        samples, weights = [], []
        for x, wx in zip((-np.sqrt(3), 0, np.sqrt(3)), (1/6, 2/3, 1/6), strict=True):
            for y, wy in zip((-np.sqrt(3), 0, np.sqrt(3)), (1/6, 2/3, 1/6), strict=True):
                sigma = np.deg2rad(geometry.acceptance_sigma_degrees)[:, None]
                ray = (directions + np.tan(x * sigma) * tangent_a
                       + np.tan(y * sigma) * tangent_b)
                ray /= np.linalg.norm(ray, axis=1, keepdims=True)
                samples.append(ray @ view.camera_to_body)
                weights.append(wx * wy)
        self.rays = np.stack(samples, axis=1)
        self.weights = np.array(weights)

    def project(self, image, *, frame_id, time, pixel_valid=None):
        response, valid, coverage = self.sample_pixels(image, pixel_valid=pixel_valid)
        return EyeSample(self.geometry.side, frame_id, response, valid, time,
                         self.geometry.sha256), coverage

    def sample_pixels(self, image, *, pixel_valid=None):
        """Geometry-only sampling shared by legacy eyes and source-specific view maps."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError("Source must be H x W x RGB")
        if image.dtype == np.uint8:
            image = image.astype(float) / 255
        elif (image.dtype.kind != "f" or not np.isfinite(image).all()
              or np.any((image < 0) | (image > 1))):
            raise ValueError("RGB must be uint8 or finite float in [0,1]")
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=bool) if pixel_valid is None else readonly_mask(pixel_valid)
        if mask.shape != (h, w):
            raise ValueError("Source validity shape differs from image")
        forward, left, up = np.moveaxis(self.rays, -1, 0)
        denominator = np.where(forward > 0, forward, 1)
        tangent = np.tan(np.deg2rad(self.view.fovy_degrees / 2))
        x = (.5 - left / denominator / (2 * tangent * w / h)) * w - .5
        y = (.5 - up / denominator / (2 * tangent)) * h - .5
        support = (forward > 0) & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
        x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
        support &= mask[y0, x0] & mask[y0, x1] & mask[y1, x0] & mask[y1, x1]
        wx, wy = (x-x0)[..., None], (y-y0)[..., None]
        pixels = (image[y0, x0]*(1-wx)*(1-wy) + image[y0, x1]*wx*(1-wy)
                  + image[y1, x0]*(1-wx)*wy + image[y1, x1]*wx*wy)
        weights = support * self.weights
        coverage = weights.sum(axis=1)
        valid = coverage >= self.min_coverage
        response = (pixels * weights[..., None]).sum(axis=1) / np.maximum(coverage[:, None], 1e-12)
        response[~valid] = 0
        return readonly_array(response), readonly_array(valid, bool), readonly_array(coverage)


def synthetic_eye(side, rows=8, columns=12, azimuth_degrees=0, width_degrees=100,
                  height_degrees=70, sigma_degrees=1, origin=(0, 0, 0)):
    """Synthetic staggered angular lattice; never labeled an anatomical eye map."""
    if type(rows) is not int or type(columns) is not int or min(rows, columns) < 1:
        raise ValueError("Synthetic eye rows/columns must be positive integers")
    if (not np.isfinite([azimuth_degrees, width_degrees, height_degrees]).all()
            or not 0 < width_degrees < 180 or not 0 < height_degrees < 180):
        raise ValueError("Synthetic eye angular extent invalid")
    row, col = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    offset = ((row % 2) - .5) * .5
    az = np.deg2rad(azimuth_degrees + ((col+.5+offset)/columns-.5)*width_degrees)
    el = np.deg2rad((.5-(row+.5)/rows)*height_degrees)
    rays = np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], axis=-1)
    return EyeGeometry(side, rays.reshape(-1, 3), np.full(rows*columns, sigma_degrees),
                       origin, "synthetic_staggered_angular_lattice_not_measured_anatomy")
