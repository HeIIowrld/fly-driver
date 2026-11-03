"""Geometry-based pooling of T4/T5-like direction votes; an engineering adapter."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .ac_native_physics import NativePhysicsSample
from .episode_storage import load_array
from .flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from .graph import sha256
from .sensor_contract import readonly_array


class StationaryMotionReference:
    """Collect a causal reference only with fresh native evidence of a stationary body."""

    def __init__(self, *, mean_frames=3):
        if type(mean_frames) is not int or not 1 <= mean_frames <= 11:
            raise ValueError('Explicit stationary mean window from 1 to 11 required')
        self.mean_frames = mean_frames
        self.reset()

    def reset(self):
        self.rows = []
        self.value = None
        self.last_capture = self.last_packet = -1

    def observe(self, features, native_sample, *, capture_id, decision_time):
        if not isinstance(native_sample, NativePhysicsSample):
            raise ValueError("Native body sample required; no pose or desired-motion reference")
        if type(capture_id) is not int or capture_id < 0 or not np.isfinite(decision_time):
            raise ValueError("Actual capture identity and finite original decision clock required")
        features = np.asarray(features)
        if features.shape != (6,) or not np.isfinite(features).all():
            raise ValueError("Six finite visual features required")
        fields = native_sample.fields
        age = decision_time - native_sample.received_monotonic_seconds
        stationary = (
            0 <= age <= 0.05
            and abs(fields["speedKmh"]) <= 0.1
            and np.max(np.abs(fields["wheelAngularSpeed"])) <= 0.05
            and np.max(np.abs(fields["localAngularVel"])) <= 0.02
        )
        if not stationary:
            self.rows.clear()
            return False
        if capture_id <= self.last_capture or native_sample.packet_id <= self.last_packet:
            return False
        if self.rows and decision_time <= self.rows[-1][0]:
            raise ValueError("Reference decision clock reversed")
        if self.rows and decision_time-self.rows[-1][0] > .15:
            self.rows.clear()
        self.last_capture, self.last_packet = capture_id, native_sample.packet_id
        self.rows.append((decision_time, features.copy()))
        if len(self.rows) >= 11 and self.rows[-1][0] - self.rows[0][0] >= 0.4 - 1e-10:
            self.value = readonly_array(np.mean([row[1] for row in self.rows[-self.mean_frames:]], axis=0))
            return True
        return False


class CalibratedVisualSelfMotion:
    """Load only a passed calibration; output synthetic motion coordinates, not a body sensor."""

    def __init__(self, directory, *, camera_config_file="configs/ac-csp-yaw-retina.json"):
        directory = Path(directory)
        receipt = json.loads((directory / "result.json").read_text(encoding="utf8"))
        if receipt["status"] != "completed" or not receipt["calibration_gate_passed"]:
            raise ValueError("Visual self-motion calibration did not pass its declared gates")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))["config"]
        if manifest["plan"].get("reference_baseline_frames") != [8, 11]:
            raise ValueError("Stationary-reference calibration contract required")
        if sha256(camera_config_file) != manifest["camera_config_sha256"]:
            raise ValueError("Camera projection differs from calibrated visual readout")
        if sha256(directory / "calibration.npz") != receipt["calibration_sha256"]:
            raise ValueError("Visual calibration changed")
        with load_array(directory / "calibration.npz", allow_pickle=False) as archive:
            self.center = archive["center"].copy()
            self.scale = archive["scale"].copy()
            self.coefficients = archive["coefficients"].copy()
        if self.center.shape != (6,) or self.coefficients.shape != (7, 6):
            raise ValueError("Declared six-coordinate affine calibration required")
        if (
            self.scale.shape != (6,)
            or np.any(self.scale <= 0)
            or not all(
                np.isfinite(array).all() for array in (self.center, self.scale, self.coefficients)
            )
        ):
            raise ValueError("Invalid visual calibration numeric values")
        self.calibration_sha256 = receipt["calibration_sha256"]

    def decode(self, features, *, reference):
        if not isinstance(reference, StationaryMotionReference) or reference.value is None:
            raise ValueError("Verified stationary visual reference is not ready")
        features = np.asarray(features)
        if features.shape != (6,) or not np.isfinite(features).all():
            raise ValueError("Six finite visual features required")
        design = np.r_[(features - reference.value - self.center) / self.scale, 1.0]
        return readonly_array(design @ self.coefficients)


class VisualSelfMotionPool:
    """Pool into rotation/translation templates without maps, depth or vehicle pose input.

    T4/T5 channel meanings follow the pinned author direction table. No cell is
    renamed or flipped to hide the known T4d contrast mismatch. Output is uncalibrated
    neural evidence until an explicitly learned linear calibration is supplied.
    """

    @classmethod
    def from_registration(cls, config, registration):
        """Check saved array ordering and geometry before interpreting motion channels.

        This checks declared identities and geometric reconstruction. It cannot prove
        the stored neuron indices actually have those types in a source connectome.
        """
        required = {'u', 'v', 'image_uv', 'body_directions', 'view_order',
                    'motion_types', 'motion_indices'}
        if not required.issubset(registration):
            raise ValueError('Complete four-view motion registration required')
        motion_types = [f'{name}{direction}' for name in ('T4', 'T5') for direction in 'abcd']
        if (registration['view_order'].tolist() != list(VIEW_ORDER)
                or registration['motion_types'].tolist() != motion_types):
            raise ValueError('Declared view or motion channel ordering differs')
        indices = np.asarray(registration['motion_indices'])
        if (indices.shape != (8, 721) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.unique(indices).size != indices.size):
            raise ValueError('Distinct nonnegative integer motion neuron indices required')
        projector = FlyvisCameraProjector(config, registration['u'], registration['v'])
        uv, rays = np.asarray(registration['image_uv']), np.asarray(registration['body_directions'])
        if (uv.shape != projector.uv.shape or rays.shape != projector.directions_body.shape
                or not np.isfinite(uv).all() or not np.isfinite(rays).all()
                or not np.allclose(uv, projector.uv, rtol=0, atol=1e-10)
                or not np.allclose(rays, projector.directions_body, rtol=0, atol=1e-10)):
            raise ValueError('Saved receptor coordinates or view rays differ from camera geometry')
        return cls(config, rays)

    @staticmethod
    def verify_motion_node_identities(registration, node_types, node_u, node_v):
        """Compare all declared T4/T5 indices and lattice positions with node arrays."""
        indices = np.asarray(registration['motion_indices'])
        types = np.asarray(node_types).astype(str)
        u, v = np.asarray(node_u), np.asarray(node_v)
        if (indices.shape != (8, 721) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or types.ndim != 1 or u.shape != types.shape
                or v.shape != types.shape or np.any(indices >= len(types))
                or np.unique(indices).size != indices.size):
            raise ValueError('Valid distinct motion indices and aligned source nodes required')
        expected_types = np.array([f'{n}{d}' for n in ('T4', 'T5') for d in 'abcd'])
        expected_u, expected_v = np.asarray(registration['u']), np.asarray(registration['v'])
        if (expected_u.shape != (721,) or expected_v.shape != (721,)
                or not np.all(types[indices] == expected_types[:, None])
                or not np.all(u[indices] == expected_u[None, :])
                or not np.all(v[indices] == expected_v[None, :])):
            raise ValueError('Motion indices do not match source neuron type and lattice position')

    def __init__(self, config, rays):
        rays = np.asarray(rays, dtype=float)
        if rays.shape != (4, 721, 3) or not np.allclose(np.linalg.norm(rays, axis=-1), 1):
            raise ValueError("Four camera-local sets of 721 unit body rays required")
        templates = []
        for index, name in enumerate(VIEW_ORDER):
            camera = config["views"][name]
            ray = rays[index]
            forward, right, up = (
                np.array(camera[key]) for key in ("forward_body", "right_body", "up_body")
            )
            depth = ray @ forward
            if np.any(depth <= 0):
                raise ValueError("Rays behind their source camera")
            directions = []
            for axis in np.eye(3):
                directions.append(np.cross(ray, axis))  # stationary scene, rotating observer
            for axis in np.eye(3):
                directions.append(-axis + (ray @ axis)[:, None] * ray)  # unit inverse-depth flow
            row = []
            for derivative in directions:
                du = (
                    0.5
                    * camera["projection_x"]
                    * ((derivative @ right) * depth - (ray @ right) * (derivative @ forward))
                    / depth**2
                )
                dv_up = (
                    0.5
                    * camera["projection_y"]
                    * ((derivative @ up) * depth - (ray @ up) * (derivative @ forward))
                    / depth**2
                )
                row.append(np.column_stack([du, dv_up]))
            templates.append(row)
        self.templates = np.array(templates)  # view, mode, receptor, screen-axis
        self.scale = np.sqrt(np.mean(self.templates**2, axis=(0, 2, 3)))

    def features(self, motion_activity):
        activity = np.asarray(motion_activity)
        if activity.shape[-3:] != (4, 8, 721) or not np.isfinite(activity).all():
            raise ValueError("Finite ordered view/T4a-d/T5a-d/retinal response required")
        activity = np.maximum(activity, 0)
        horizontal = activity[..., :, 1, :] - activity[..., :, 0, :]
        horizontal += activity[..., :, 5, :] - activity[..., :, 4, :]
        upward = activity[..., :, 2, :] - activity[..., :, 3, :]
        upward += activity[..., :, 6, :] - activity[..., :, 7, :]
        votes = np.stack([horizontal, upward], axis=-1)
        pooled = np.einsum("...vra,vmra->...m", votes, self.templates) / (4 * 721)
        return pooled / self.scale


class SphereCameraStimulus:
    """Procedural static textured sphere viewed by separate moving camera origins.

    Only controlled component stimuli, not an AC road or biological environment.
    Translation target is velocity divided by the declared sphere radius.
    """

    def __init__(self, config, camera_origins_body, *, radius=20.0):
        self.origins = np.asarray(camera_origins_body, dtype=np.float32)
        if self.origins.shape != (4, 3) or not np.isfinite(self.origins).all() or radius <= 5:
            raise ValueError("Separate finite camera origins and explicit sphere radius required")
        y, x = np.mgrid[:180, :320]
        directions = []
        for name in VIEW_ORDER:
            camera = config["views"][name]
            ray = (
                np.array(camera["forward_body"])
                + (2 * (x[..., None] + 0.5) / 320 - 1)
                / camera["projection_x"]
                * np.array(camera["right_body"])
                + (1 - 2 * (y[..., None] + 0.5) / 180)
                / camera["projection_y"]
                * np.array(camera["up_body"])
            )
            directions.append(ray / np.linalg.norm(ray, axis=-1, keepdims=True))
        self.rays = np.array(directions, dtype=np.float32)
        self.radius = float(radius)

    def render(self, angular_body, normalized_translation_body, time, *, texture_seed):
        omega = np.asarray(angular_body)
        velocity = np.asarray(normalized_translation_body) * self.radius
        rotation = Rotation.from_rotvec(omega * time).as_matrix().astype(np.float32)
        rays = self.rays @ rotation.T
        origins = self.origins @ rotation.T + velocity * time
        projection = np.sum(rays * origins[:, None, None, :], axis=-1)
        discriminant = projection**2 + self.radius**2 - np.sum(origins**2, axis=-1)[:, None, None]
        if np.any(discriminant <= 0):
            raise ValueError("Observer left synthetic sphere")
        distance = -projection + np.sqrt(discriminant)
        point = (origins[:, None, None, :] + distance[..., None] * rays) / self.radius
        random = np.random.default_rng(texture_seed)
        waves = random.normal(size=(4, 3)).astype(np.float32)
        waves /= np.linalg.norm(waves, axis=-1, keepdims=True)
        waves *= np.array([12.0, 21.0, 33.0, 47.0], dtype=np.float32)[:, None]
        phases = random.uniform(-np.pi, np.pi, 4).astype(np.float32)
        intensity = 0.5 + 0.12 * np.sin(point @ waves.T + phases).sum(axis=-1)
        pixels = np.rint(np.clip(intensity, 0, 1) * 255).astype(np.uint8)
        return np.repeat(pixels[..., None], 3, axis=-1)
