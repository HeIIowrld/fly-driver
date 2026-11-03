"""MuJoCo 3D prototype. O: oracle diagnostics, V: cameras + instrumented body."""

from dataclasses import asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .control import IntentController, VehicleConfig
from .road import RoadConfig, SyntheticRoad
from .vision import CameraConfig, MotionFilters, RaySampler

BODY_CHANNELS = (
    "gyro_x gyro_y gyro_z accel_x accel_y accel_z "
    "susp_fl susp_fr susp_rl susp_rr dsusp_fl dsusp_fr dsusp_rl dsusp_rr "
    "wheel_fl wheel_fr wheel_rl wheel_rr steer_fl steer_fr "
    "command_torque_l command_torque_r previous_speed previous_curvature encoder_speed "
    "spring_load_fl spring_load_fr spring_load_rl spring_load_rr "
    "gravity_est_x gravity_est_y gravity_est_z "
    "oracle_progress oracle_lateral oracle_heading oracle_grade oracle_bank oracle_curvature"
).split()
BODY_SCALE = np.array(
    [5] * 3
    + [20] * 3
    + [0.07] * 4
    + [1] * 4
    + [50] * 4
    + [0.5] * 2
    + [160] * 2
    + [1] * 2
    + [8]
    + [1000] * 4
    + [1] * 3
    + [1, 3, np.pi, 0.2, 0.3, 0.3],
    dtype=np.float32,
)


def model_xml(road, vehicle, timestep, camera=None):
    c = vehicle
    camera = camera or CameraConfig()
    wheels = []
    for i, name in enumerate(("fl", "fr", "rl", "rr")):
        x = c.wheelbase / 2 if i < 2 else -c.wheelbase / 2
        y = c.track / 2 if i % 2 == 0 else -c.track / 2
        steer_joint = (
            f'<joint name="steer_{name}" type="hinge" axis="0 0 1" range="-.5 .5" damping="5"/>'
            if i < 2
            else ""
        )
        wheels.append(f'''
          <body name="susp_{name}" pos="{x} {y} -.28">
            <inertial pos="0 0 0" mass=".5" diaginertia=".005 .005 .005"/>
            <joint name="susp_{name}" type="slide" axis="0 0 1" range="-.07 .07"
              stiffness="{c.suspension_stiffness}" damping="{c.suspension_damping}"/>
            <body name="steering_{name}">
              <inertial pos="0 0 0" mass=".2" diaginertia=".002 .002 .002"/>
              {steer_joint}
              <body name="wheel_{name}">
                <joint name="spin_{name}" type="hinge" axis="0 1 0" damping=".1"/>
                <geom name="wheel_{name}" type="cylinder" size="{c.wheel_radius} .06"
                  quat=".70710678 .70710678 0 0" mass="8" rgba=".1 .1 .1 1"
                  friction="1 .005 .0001" condim="6"/>
              </body>
            </body>
          </body>''')
    cameras = []
    for name, side in (("left", 1), ("right", -1)):
        theta, pitch = np.deg2rad(side * camera.yaw_degrees), np.deg2rad(camera.pitch_degrees)
        right = np.array([np.sin(theta), -np.cos(theta), 0.0])
        up = np.array([-np.sin(pitch) * np.cos(theta),
                       -np.sin(pitch) * np.sin(theta), np.cos(pitch)])
        cameras.append(
            f'<camera name="{name}" pos="{camera.forward} {side * camera.baseline / 2} '
            f'{camera.height}" fovy="{camera.fovy}" '
            f'xyaxes="{" ".join(str(v) for v in np.r_[right, up])}"/>'
        )
    r = road.config
    return f'''<mujoco model="mecar_flybrain_prototype">
      <compiler angle="radian"/>
      <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <visual><global offwidth="{max(128, camera.source_size)}"
        offheight="{max(128, camera.source_size)}"/></visual>
      <asset><hfield name="road" nrow="161" ncol="481"
        size="{(r.length + 20) / 2} 20 1 .2"/></asset>
      <worldbody>
        <light pos="0 0 30" dir="0 0 -1" directional="true"/>
        <geom name="terrain" type="hfield" hfield="road" pos="{r.length / 2} 0 0"
          rgba=".35 .4 .3 1" friction="1 .005 .0001"/>
        <body name="chassis" pos="0 0 .5"><freejoint name="root"/>
          <geom name="chassis" type="box" size=".85 .42 .1" mass="{c.chassis_mass}"
            rgba=".15 .3 .8 1"/>
          <site name="imu" size=".01" pos="0 0 0"/>
          {"".join(cameras)}{"".join(wheels)}
        </body>
      </worldbody>
      <actuator>
        <position name="steer_fl" joint="steer_fl" kp="1500" kv="80"
          ctrllimited="true" ctrlrange="-.5 .5"/>
        <position name="steer_fr" joint="steer_fr" kp="1500" kv="80"
          ctrllimited="true" ctrlrange="-.5 .5"/>
        <motor name="drive_l" joint="spin_rl"/>
        <motor name="drive_r" joint="spin_rr"/>
        {"".join(f'<motor name="brake_{n}" joint="spin_{n}"/>' for n in ("fl", "fr", "rl", "rr"))}
      </actuator>
      <sensor><gyro name="gyro" site="imu"/><accelerometer name="accel" site="imu"/></sensor>
    </mujoco>'''


class FlyBrainEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mode="O",
        road=None,
        vehicle=None,
        max_steps=1000,
        image_size=32,
        control_dt=0.01,
        physics_dt=0.001,
        camera_hz=30,
        sensor_profile="full",
        actuation_mode="legacy",
        camera=None,
        visual_schema="legacy",
        road_visual="edge_markers",
    ):
        super().__init__()
        import mujoco

        self.mj = mujoco
        if mode not in ("O", "V"):
            raise ValueError("Only O diagnostics and V sensors are implemented; M is planned")
        ratio = control_dt / physics_dt
        if control_dt <= 0 or physics_dt <= 0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be a positive integer multiple of physics_dt")
        if camera_hz <= 0 or max_steps <= 0 or image_size < 16:
            raise ValueError("Invalid camera/episode configuration")
        self.mode = mode
        if road_visual not in ("edge_markers", "lane_surface"):
            raise ValueError("Unknown road_visual")
        self.road_visual = road_visual
        if sensor_profile not in ("full", "no_proprioception", "no_efference_copy"):
            raise ValueError("Unknown sensor_profile")
        self.sensor_profile = sensor_profile
        self.road = SyntheticRoad(RoadConfig(**(road or {})))
        self.vehicle = VehicleConfig(**(vehicle or {}))
        self.control = IntentController(self.vehicle, actuation_mode)
        self.camera = CameraConfig(**(camera or {}))
        if visual_schema not in ("legacy", "retina_v1"):
            raise ValueError("Unknown visual_schema")
        if self.camera.sampling.startswith("angular_") and visual_schema != "retina_v1":
            raise ValueError("Angular sampling requires the retina_v1 mask contract")
        self.visual_schema = visual_schema
        self.sampler = RaySampler(self.camera, image_size)
        self.spatial_valid = np.repeat(self.sampler.valid[None], 2, axis=0)
        self.motion_filters = MotionFilters(
            self.spatial_valid, staggered=self.camera.sampling == "angular_hex"
        )
        self.dt, self.substeps, self.max_steps = control_dt, round(ratio), max_steps
        self.image_size, self.camera_period = image_size, 1 / camera_hz
        self.model = mujoco.MjModel.from_xml_string(
            model_xml(self.road, self.vehicle, physics_dt, self.camera)
        )
        self.render_scene_option = mujoco.MjvOption()
        if road_visual == "lane_surface":
            # Controlled visual baseline: avoid software shadow-map acne on thin paint.
            self.model.light_castshadow[:] = False
        self._flat_visual_ground = road_visual == "lane_surface" and self.road.config.flattened
        if self._flat_visual_ground:
            # Keep the collision heightfield exactly; render an equivalent flat plane.
            # The dense flat heightfield otherwise wastes software-renderer work.
            self.model.geom_group[self.model.geom("terrain").id] = 5
            self.render_scene_option.geomgroup[5] = 0
        # MuJoCo visual-map distances multiply model.stat.extent. A 100 m
        # heightfield made the default near plane about 1.27 m, clipping the
        # nearby road out of wide cameras. New configs declare distances in m.
        if self.camera.near_m:
            self.model.vis.map.znear = self.camera.near_m / self.model.stat.extent
            self.model.vis.map.zfar = self.camera.far_m / self.model.stat.extent
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self._prepare_surface()
        self.susp_joints = [self.model.joint("susp_" + n) for n in ("fl", "fr", "rl", "rr")]
        self.spin_joints = [self.model.joint("spin_" + n) for n in ("fl", "fr", "rl", "rr")]
        self.steer_joints = [self.model.joint("steer_" + n) for n in ("fl", "fr")]
        self.action_space = spaces.Box(-1, 1, (self.control.action_size,), np.float32)
        self.observation_space = spaces.Dict(
            {
                "body": spaces.Box(-100, 100, (38,), np.float32),
                "body_valid": spaces.Box(0, 1, (38,), np.float32),
                "images": spaces.Box(0, 255, (2, image_size, image_size, 3), np.uint8),
                "camera_valid": spaces.Box(0, 1, (2,), np.float32),
                "camera_age": spaces.Box(0, 1e6, (2,), np.float32),
            }
        )
        if visual_schema == "retina_v1":
            self.observation_space.spaces.update(
                {
                    "retina_valid": spaces.Box(0, 1, (2, image_size, image_size), np.float32),
                    "motion": spaces.Box(0, 1, (2, image_size, image_size, 6), np.float32),
                    "motion_valid": spaces.Box(0, 1, (2, image_size, image_size, 6), np.float32),
                    "previous_yaw_action": spaces.Box(-1, 1, (1,), np.float32),
                    "camera_frame_id": spaces.Box(0, 2**60, (2,), np.int64),
                }
            )
        self.active = False

    def _prepare_surface(self):
        r = self.road.config
        x = np.linspace(-10, r.length + 10, 481)
        y = np.linspace(-20, 20, 161)
        heights = self.road.height(x[None, :], y[:, None])
        low, high = float(heights.min()), float(heights.max())
        amplitude = max(high - low, 0.001)
        self.model.hfield_size[0, 2] = amplitude
        self.model.hfield_data[:] = ((heights - low) / amplitude).ravel()
        self.model.geom_pos[self.model.geom("terrain").id, 2] = low
        # Visible road-edge markers share the same heightfield, no contact geometry.
        # Added at render time as scene geoms; they are not hidden oracle actor inputs.
        self._markers = []
        for marker_x in np.arange(0, r.length, 2):
            frame = self.road.frame(float(marker_x))
            for side in (-1, 1):
                point = np.array([marker_x, self.road.center_y(marker_x), 0.0])
                point += frame[:, 1] * side * r.width / 2
                point[2] = self.road.height(point[0], point[1]) + 0.07
                color = [1.0, 0.8, 0.05, 1.0] if side > 0 else [0.05, 0.3, 1.0, 1.0]
                self._markers.append((point, np.array(color, dtype=np.float32)))
        self._marker_size = np.array([0.07] * 3)
        self._marker_rotation = np.eye(3).ravel()
        self._road_visuals = []
        if self.road_visual == "lane_surface":
            # Static visible scenery only. No new collision geoms or actor truth fields.
            # Tessellated tangent strips approximate paint/asphalt on the heightfield.
            self._markers = []
            for x0 in np.arange(0, r.length, .5):
                x1 = min(x0 + .5, r.length)
                x = (x0 + x1) / 2
                y = self.road.center_y(x)
                frame = self.road.frame(float(x))
                center = np.array([x, y, self.road.height(x, y)])
                length = (x1 - x0) / frame[0, 0]
                for side, width, lift, color in (
                    (0, r.width / 2, .012, [.16, .16, .16, 1]),
                    (-1, .06, .018, [.95, .95, .95, 1]),
                    (1, .06, .018, [.95, .95, .95, 1]),
                ):
                    point = (center + frame[:, 1] * side * (r.width / 2 - .07)
                             + frame[:, 2] * lift)
                    self._road_visuals.append((
                        np.array([length / 2 + .002, width, .002]), point,
                        frame.ravel(), np.array(color, dtype=np.float32),
                    ))

    def _joint_positions(self, joints):
        return np.array([self.data.qpos[j.qposadr[0]] for j in joints])

    def _joint_velocities(self, joints):
        return np.array([self.data.qvel[j.dofadr[0]] for j in joints])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.mj.mj_resetData(self.model, self.data)
        start_x = float((options or {}).get("start_x", 3.0))
        if not 0 <= start_x < self.road.config.length - 5:
            raise ValueError("start_x outside synthetic course")
        y = float(self.road.center_y(start_x))
        frame = self.road.frame(start_x)
        lateral = float((options or {}).get("lateral_offset", 0.0))
        heading = float((options or {}).get("heading_offset_degrees", 0.0))
        if not np.all(np.isfinite([lateral, heading])) or (
            abs(lateral) > 0.4 * self.road.config.width or abs(heading) > 30
        ):
            raise ValueError("Initial pose perturbation exceeds prototype domain")
        point = np.array([start_x, y, float(self.road.height(start_x, y)) + 0.53])
        if lateral:
            point += frame[:, 1] * lateral
            point[2] = self.road.height(point[0], point[1]) + 0.53
        if heading:
            angle = np.deg2rad(heading)
            frame = frame @ np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0], [0, 0, 1],
            ])
        q = np.zeros(4)
        self.mj.mju_mat2Quat(q, frame.ravel())
        self.data.qpos[:3] = point
        self.data.qpos[3:7] = q
        self.mj.mj_forward(self.model, self.data)
        # Settle compliant suspension/contact with sensor-based viscous brakes.
        for _ in range(round(0.8 / self.model.opt.timestep)):
            self.data.ctrl[4:] = -100 * np.tanh(self._joint_velocities(self.spin_joints) / 0.5)
            self.mj.mj_step(self.model, self.data)
        self.control.reset()
        self.data.ctrl[:] = 0
        self.mj.mj_forward(self.model, self.data)
        self.steps = 0
        self.previous_action = np.zeros(self.control.action_size)
        self.gravity_estimate = np.array([0.0, 0.0, -1.0])
        self.images = np.zeros((2, self.image_size, self.image_size, 3), np.uint8)
        self.camera_source_images = np.zeros(
            (2, self.sampler.source_size, self.sampler.source_size, 3), np.uint8
        )
        self.last_exposure = -np.inf
        self.exposure_id = 0
        self.motion_filters.reset()
        self.motion = np.zeros((2, self.image_size, self.image_size, 6), np.float32)
        self.motion_valid = np.zeros_like(self.motion)
        self.command_info = {"rear_torque": [0.0, 0.0]}
        self.active = True
        info = self._info("running")
        self.last_progress = info["truth"]["progress_3d"]
        return self._observation(info["truth"]), info

    def _observation(self, truth=None):
        gyro = self.data.sensor("gyro").data.copy()
        accel = self.data.sensor("accel").data.copy()
        susp = self._joint_positions(self.susp_joints)
        dsusp = self._joint_velocities(self.susp_joints)
        omega = self._joint_velocities(self.spin_joints)
        speed = float(np.mean(omega[:2]) * self.vehicle.wheel_radius)
        load_est = np.maximum(
            0, self.vehicle.suspension_stiffness * susp + self.vehicle.suspension_damping * dsusp
        )
        # Provisional complementary estimate; trust accelerometer only near rest.
        # Acceleration during sustained turns requires the planned full estimator.
        self.gravity_estimate -= np.cross(gyro, self.gravity_estimate) * self.dt
        if abs(speed) < 0.2 and 8.5 < np.linalg.norm(accel) < 11.0:
            self.gravity_estimate = 0.95 * self.gravity_estimate - 0.05 * accel / np.linalg.norm(
                accel
            )
        self.gravity_estimate /= max(np.linalg.norm(self.gravity_estimate), 1e-6)
        body = np.r_[
            gyro,
            accel,
            susp,
            dsusp,
            omega,
            self._joint_positions(self.steer_joints),
            self.command_info["rear_torque"],
            self.previous_action[:2],
            speed,
            load_est,
            self.gravity_estimate,
            np.zeros(6),
        ].astype(np.float32)
        valid = np.ones(38, dtype=np.float32)
        if self.mode == "O":
            if truth is None:
                s, ey, point, frame = self.road.project(self.data.qpos[:3])
            else:
                s, ey = truth["progress_3d"], truth["lateral_error"]
                point, frame = np.asarray(truth["road_point"]), np.asarray(truth["road_frame"])
            rotation = self.data.xmat[self.model.body("chassis").id].reshape(3, 3)
            heading = np.arctan2(
                np.dot(rotation[:, 0], frame[:, 1]), np.dot(rotation[:, 0], frame[:, 0])
            )
            tangent = frame[:, 0]
            bank = np.arctan2(frame[2, 1], np.linalg.norm(frame[:2, 1]))
            f_next = self.road.frame(min(float(point[0]) + 0.1, self.road.config.length))
            curvature = np.dot((f_next[:, 0] - tangent) / 0.1, frame[:, 1])
            body[-6:] = [
                s / self.road.s[-1],
                ey,
                heading,
                tangent[2] / np.linalg.norm(tangent[:2]),
                bank,
                curvature,
            ]
        else:
            valid[-6:] = 0
        if self.sensor_profile == "no_proprioception":
            # Keep issued commands/actions (efference copy), mask measured body sensing.
            valid[:20] = 0
            valid[24:32] = 0
            body[valid == 0] = 0
        elif self.sensor_profile == "no_efference_copy":
            # Remove policy command history; keep measured wheel/steering/body signals.
            valid[20:24] = 0
            body[20:24] = 0
        camera_valid = np.zeros(2, dtype=np.float32)
        age = np.zeros(2, dtype=np.float32)
        if self.mode == "V":
            if self.data.time - self.last_exposure >= self.camera_period - 1e-9:
                self.images = self._render_cameras()
                self.last_exposure = float(self.data.time)
                self.exposure_id += 1
                if self.visual_schema == "retina_v1":
                    self.motion, self.motion_valid = self.motion_filters.update(
                        self.images, self.last_exposure
                    )
            camera_valid[:] = 1
            age[:] = self.data.time - self.last_exposure
        observation = {
            "body": np.clip(body / BODY_SCALE, -100, 100),
            "body_valid": valid,
            "images": self.images.copy(),
            "camera_valid": camera_valid,
            "camera_age": age,
        }
        if self.visual_schema == "retina_v1":
            yaw_action = self.previous_action[2] if self.control.actuation_mode == "residual" else 0
            if self.sensor_profile == "no_efference_copy":
                yaw_action = 0
            observation.update(
                retina_valid=self.spatial_valid.copy(),
                motion=self.motion.copy(),
                motion_valid=self.motion_valid.copy(),
                previous_yaw_action=np.array([yaw_action], np.float32),
                camera_frame_id=np.full(2, self.exposure_id, np.int64),
            )
        return observation

    def _render_cameras(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(
                self.model, self.sampler.source_size, self.sampler.source_size
            )
        frames = []
        sources = []
        for name in ("left", "right"):
            self.renderer.update_scene(self.data, camera=name,
                                       scene_option=self.render_scene_option)
            self.add_render_markers(self.renderer.scene)
            source = self.renderer.render().copy()
            sources.append(source)
            frames.append(self.sampler.sample(source))
        self.camera_source_images = np.stack(sources)
        return np.stack(frames)

    def add_render_markers(self, scene):
        """Static visible road scenery shared by all rendering observers."""
        if self._flat_visual_ground:
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Renderer scene capacity exceeded")
            self.mj.mjv_initGeom(scene.geoms[scene.ngeom], self.mj.mjtGeom.mjGEOM_PLANE,
                                 np.array([self.road.config.length / 2 + 10, 20., .1]),
                                 np.array([self.road.config.length / 2, 0., 0.]), np.eye(3).ravel(),
                                 np.array([.35, .4, .3, 1], dtype=np.float32))
            scene.ngeom += 1
        for size, point, rotation, color in self._road_visuals:
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Renderer scene capacity exceeded")
            self.mj.mjv_initGeom(scene.geoms[scene.ngeom], self.mj.mjtGeom.mjGEOM_BOX,
                                 size, point, rotation, color)
            scene.ngeom += 1
        for point, color in self._markers:
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Renderer scene capacity exceeded")
            self.mj.mjv_initGeom(
                scene.geoms[scene.ngeom],
                self.mj.mjtGeom.mjGEOM_SPHERE,
                self._marker_size,
                point,
                self._marker_rotation,
                color,
            )
            scene.ngeom += 1

    def _info(self, reason):
        s, ey, point, frame = self.road.project(self.data.qpos[:3])
        return {
            "truth": {
                "position_xyz": self.data.qpos[:3].tolist(),
                "quaternion_wxyz": self.data.qpos[3:7].tolist(),
                "progress_3d": s,
                "lateral_error": ey,
                "road_point": point.tolist(),
                "road_frame": frame.tolist(),
            },
            "time": float(self.data.time),
            "reason": reason,
            "mode": self.mode,
            "sensor_profile": self.sensor_profile,
            "control": self.command_info,
            "road_provenance": self.road.provenance,
        }

    def step(self, action):
        if not self.active:
            raise RuntimeError("reset is required after episode end")
        omega = self._joint_velocities(self.spin_joints)
        speed = float(np.mean(omega[:2]) * self.vehicle.wheel_radius)
        yaw_rate = float(self.data.sensor("gyro").data[2])
        self.data.ctrl[:], self.command_info = self.control.command(
            action, speed, yaw_rate, omega, self.dt
        )
        for _ in range(self.substeps):
            self.mj.mj_step(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        self.steps += 1
        self.previous_action = np.clip(np.asarray(action, dtype=float), -1, 1)
        info = self._info("running")
        s, ey = info["truth"]["progress_3d"], info["truth"]["lateral_error"]
        rotation = self.data.xmat[self.model.body("chassis").id].reshape(3, 3)
        up = np.array(info["truth"]["road_frame"])[:, 2]
        reason = "running"
        if not np.all(np.isfinite(self.data.qpos)):
            reason = "numerical_failure"
        elif np.dot(rotation[:, 2], up) < 0.4:
            reason = "overturn"
        elif abs(ey) > self.road.config.width / 2:
            reason = "track_exit"
        elif s >= self.road.s[-1] - 1:
            reason = "completion"
        else:
            chassis_id = self.model.geom("chassis").id
            if any(
                chassis_id in (contact.geom1, contact.geom2)
                for contact in self.data.contact[: self.data.ncon]
            ):
                reason = "chassis_collision"
        terminated = reason != "running"
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            reason = "time_limit"
        reward = float(s - self.last_progress - 0.02 * ey**2 * self.dt)
        if terminated and reason != "completion":
            reward -= 5
        self.last_progress = s
        info["reason"] = reason
        self.active = not (terminated or truncated)
        return self._observation(info["truth"]), reward, terminated, truncated, info

    def configuration(self):
        return {
            "mode": self.mode,
            "sensor_profile": self.sensor_profile,
            "road": asdict(self.road.config),
            "vehicle": asdict(self.vehicle),
            "max_steps": self.max_steps,
            "image_size": self.image_size,
            "control_dt": self.dt,
            "physics_dt": float(self.model.opt.timestep),
            "camera_hz": 1 / self.camera_period,
            "effective_camera_hz": 1 / (np.ceil(self.camera_period / self.dt - 1e-9) * self.dt),
            "camera": asdict(self.camera),
            "visual_schema": self.visual_schema,
            "road_visual": self.road_visual,
            "flat_visual_ground": self._flat_visual_ground,
            "actuation_mode": self.control.actuation_mode,
            "render_clip_m": {
                "near": float(self.model.vis.map.znear * self.model.stat.extent),
                "far": float(self.model.vis.map.zfar * self.model.stat.extent),
            },
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
