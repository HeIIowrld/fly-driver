"""Adapt instrumented MuJoCo observations and raw camera sources, excluding route truth."""

import numpy as np

from .compound_eye import CompoundEyeProjector, EyeGeometry, PerspectiveView, synthetic_eye
from .sensor_contract import ActorObservation, BodySample, SampleTime


class MuJoCoSensoryBridge:
    def __init__(self, env, rows=None, columns=None, sigma_degrees=1, geometry_path=None,
                 head_to_body_rotation=None):
        if env.mode != "V":
            raise ValueError("Sensory bridge requires V observations, never oracle O")
        self.projectors = {}
        anatomical = None
        if geometry_path is not None:
            from .eye_anatomy import load_anatomical_eyes

            if rows is not None or columns is not None or head_to_body_rotation is None:
                raise ValueError("Imported axes need an explicit head pose and no synthetic grid")
            origins = {side: [env.camera.forward, sign * env.camera.baseline / 2, env.camera.height]
                       for side, sign in (("left", 1), ("right", -1))}
            anatomical = {eye.side: eye for eye in load_anatomical_eyes(
                geometry_path, origins, sigma_degrees=sigma_degrees,
                head_to_body_rotation=head_to_body_rotation)}
        elif head_to_body_rotation is not None:
            raise ValueError("Head-pose override is only supported with imported axes")
        rows = 8 if rows is None else rows
        columns = 12 if columns is None else columns
        for side, sign in (("left", 1), ("right", -1)):
            theta = np.deg2rad(sign * env.camera.yaw_degrees)
            pitch = np.deg2rad(env.camera.pitch_degrees)
            forward = [np.cos(pitch)*np.cos(theta), np.cos(pitch)*np.sin(theta), np.sin(pitch)]
            left = [-np.sin(theta), np.cos(theta), 0]
            up = [-np.sin(pitch)*np.cos(theta), -np.sin(pitch)*np.sin(theta), np.cos(pitch)]
            rotation = np.column_stack([forward, left, up])
            origin = [env.camera.forward, sign * env.camera.baseline / 2, env.camera.height]
            view = PerspectiveView(rotation, origin, env.camera.fovy)
            if anatomical is None:
                local = synthetic_eye(side, rows, columns, width_degrees=env.camera.angular_width,
                                      height_degrees=env.camera.angular_height,
                                      sigma_degrees=sigma_degrees, origin=origin)
                geometry = EyeGeometry(side, local.directions_body @ rotation.T,
                                       local.acceptance_sigma_degrees, origin, local.provenance)
            else:
                geometry = anatomical[side]
            self.projectors[side] = CompoundEyeProjector(geometry, view)
        self.episode = None
        self.exposures = {}

    @property
    def eyes(self):
        return tuple(projector.geometry for projector in self.projectors.values())

    def observe(self, env, observation, *, episode_id, sequence_id, received_monotonic_seconds):
        from .env import BODY_SCALE

        if env.mode != "V":
            raise ValueError("Cannot switch sensory bridge to oracle observations")
        body, valid = np.asarray(observation["body"]), np.asarray(observation["body_valid"])
        if body.shape != (38,) or valid.shape != (38,) or np.any(body[-6:]) or np.any(valid[-6:]):
            raise ValueError("Legacy V observation must have masked oracle channels")
        raw = body * BODY_SCALE
        now = SampleTime(float(env.data.time), received_monotonic_seconds)
        samples = tuple(
            BodySample(name, raw[region], valid[region], now, unit, frame, "declared_sensor_model")
            for name, region, unit, frame in (
                ("angular_velocity", slice(0, 3), "rad/s", "body"),
                ("specific_force", slice(3, 6), "m/s2", "body"),
                ("suspension_position", slice(6, 10), "m", "compression_positive"),
                ("suspension_velocity", slice(10, 14), "m/s", "compression_positive"),
                ("wheel_angular_velocity", slice(14, 18), "rad/s", "wheel_axles"),
                ("steering_angle", slice(18, 20), "rad", "roadwheel"),
            )
        )
        if self.episode != episode_id:
            self.exposures = {}
            self.episode = episode_id
        eyes = []
        for index, (side, projector) in enumerate(self.projectors.items()):
            if not observation["camera_valid"][index]:
                # Availability loss is a missing sample, not a modified old exposure.
                continue
            previous = self.exposures.get(side)
            if previous is None or previous.frame_id != env.exposure_id:
                eye, _ = projector.project(
                    env.camera_source_images[index], frame_id=int(env.exposure_id),
                    time=SampleTime(float(env.last_exposure), received_monotonic_seconds),
                )
                self.exposures[side] = eye
            eye = self.exposures[side]
            eyes.append(eye)
        # Legacy intent/torque is not AC steering/throttle/brake; leave it unavailable.
        return ActorObservation(episode_id, sequence_id, now.simulation_seconds,
                                received_monotonic_seconds, tuple(eyes), samples)
