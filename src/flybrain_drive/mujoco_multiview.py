"""Direct calibrated view rendering for the MuJoCo component environment, not AC mirrors."""

import numpy as np


class MuJoCoMultiViewRenderer:
    """Temporarily reuse a fixed body-mounted camera; restore all camera parameters.

    Physics is not advanced while the requested perspectives are rendered. These
    views share the exact current simulation state but may be requested on separate
    decisions by the caller. No depth-based reprojection or mirror emulation occurs.
    """

    def __init__(self, env, views, source_size=96):
        if env.mode != "V" or type(source_size) is not int or not 16 <= source_size <= 512:
            raise ValueError("V mode and bounded source size required")
        self.env = env
        self.views = {v.source_id: v for v in views}
        if not self.views or len(self.views) != len(views):
            raise ValueError("Unique perspective source IDs required")
        self.camera = env.model.camera("left").id
        if env.model.cam_mode[self.camera] != env.mj.mjtCamLight.mjCAMLIGHT_FIXED:
            raise ValueError("Explicit fixed body camera required")
        self.renderer = env.mj.Renderer(env.model, source_size, source_size)

    def render(self, source_ids):
        if len(source_ids) != len(set(source_ids)) or set(source_ids) - set(self.views):
            raise ValueError("Known unique view sources required")
        env, camera = self.env, self.camera
        saved = (env.model.cam_pos[camera].copy(), env.model.cam_quat[camera].copy(),
                 float(env.model.cam_fovy[camera]))
        images = {}
        try:
            for name in source_ids:
                view = self.views[name].perspective
                rotation = view.camera_to_body
                # MuJoCo camera axes are right, up, back; our pinhole is forward, left, up.
                camera_rotation = np.column_stack([
                    -rotation[:, 1], rotation[:, 2], -rotation[:, 0]])
                quaternion = np.empty(4)
                env.mj.mju_mat2Quat(quaternion, camera_rotation.ravel())
                env.model.cam_pos[camera] = view.origin_body_m
                env.model.cam_quat[camera] = quaternion
                env.model.cam_fovy[camera] = view.fovy_degrees
                env.mj.mj_camlight(env.model, env.data)
                self.renderer.update_scene(env.data, camera="left",
                                           scene_option=env.render_scene_option)
                env.add_render_markers(self.renderer.scene)
                images[name] = self.renderer.render().copy()
        finally:
            env.model.cam_pos[camera] = saved[0]
            env.model.cam_quat[camera] = saved[1]
            env.model.cam_fovy[camera] = saved[2]
            env.mj.mj_camlight(env.model, env.data)
        return images

    def close(self):
        self.renderer.close()
