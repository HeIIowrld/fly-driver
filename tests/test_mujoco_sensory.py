import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from flybrain_drive.body_organs import BodyOrganEncoder, PositionOrganSpec
from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.mujoco_sensory import MuJoCoSensoryBridge
from flybrain_drive.policy import RecurrentPolicy
from flybrain_drive.sensory_policy import ObservationPacker, SensoryPolicyRunner
from flybrain_drive.sensory_runtime import make_sensory_runtime


def test_imported_axes_keep_head_frame_and_mask_rear_rays_and_padding(tmp_path):
    from flybrain_drive.graph import sha256

    path = tmp_path / "axes.npz"
    axes = np.array([[1., 0, 0], [-1, 0, 0], [0, 0, 1]])
    np.savez_compressed(path, left_directions_head=axes, right_directions_head=axes[:2])
    path.with_suffix(".json").write_text(json.dumps({
        "geometry_sha256": sha256(path), "upstream_commit": "test",
        "source_kind": "toy_test_not_anatomical_measurement"}), encoding="utf8")
    eye = {"geometry_path": str(path), "head_to_body_rotation": np.eye(3).tolist(),
           "sigma_degrees": 1}
    config = {"sensory_preprocessing": {"eye": eye, "visual_encoder": "global_points"}}
    env = FlyBrainEnv(mode="V", max_steps=2, camera={"fovy": 120, "source_size": 32})
    try:
        obs, _ = env.reset(seed=17)
        bridge, packer, sealed = make_sensory_runtime(env, config)
        np.testing.assert_array_equal(bridge.eyes[0].directions_body, axes)
        actor = bridge.observe(env, obs, episode_id="axes", sequence_id=0,
                               received_monotonic_seconds=100)
        packed = packer.pack(actor)
        assert packed["ray_values"].shape == (2, 3, 3)
        assert packed["ray_valid"][0, 0] and packed["ray_valid"][1, 0]
        assert not packed["ray_valid"][:, 1:].any()
        assert not packed["ray_values"][:, 1:].any()
        sealed["sensory_encoder"]["eye_map_hashes"]["left"] = "old-synthetic-map"
        with pytest.raises(ValueError, match="calibration"):
            make_sensory_runtime(env, sealed)
        with pytest.raises(ValueError, match="irregular"):
            make_sensory_runtime(env, {"sensory_preprocessing": {
                "eye": eye, "visual_encoder": "retinotopic_grid"}})
        with pytest.raises(ValueError, match="explicit head pose"):
            MuJoCoSensoryBridge(env, geometry_path=path)
        with pytest.raises(ValueError, match="no synthetic grid"):
            MuJoCoSensoryBridge(env, rows=3, **eye)
    finally:
        env.close()


def test_actual_rendered_source_and_body_feed_actor_without_route_truth():
    env = FlyBrainEnv(mode="V", max_steps=12, camera={"fovy": 120, "source_size": 32})
    try:
        obs, info = env.reset(seed=17)
        bridge = MuJoCoSensoryBridge(env, rows=3, columns=4)
        organs = BodyOrganEncoder([
            PositionOrganSpec("steering_angle", 2, -.5, .5, .15, 1),
            PositionOrganSpec("suspension_position", 4, -.07, .07, .02, 1),
        ], rotation_scale=5)
        packer = ObservationPacker(bridge.eyes, organs,
                                   specific_force_scale=20, wheel_speed_scale=50)
        policy = RecurrentPolicy(sensory_encoder={
            "organ_size": packer.organ_size,
            "eye_map_hashes": {eye.side: eye.sha256 for eye in bridge.eyes},
        })
        runner = SensoryPolicyRunner(policy, packer)
        previous_eye = None
        for tick in range(10):
            camera_input = dict(obs)
            if tick == 2:
                camera_input["camera_valid"] = np.array([0, 1], dtype=np.float32)
            actor = bridge.observe(env, camera_input, episode_id="diagnostic", sequence_id=tick,
                                   received_monotonic_seconds=100+tick)
            assert actor.previous_applied is None
            assert all(b.name not in ("route_progress", "lateral_error") for b in actor.body)
            assert actor.eyes[0].valid.any()
            if tick == 1:
                assert actor.eyes[0].frame_id == previous_eye.frame_id
                np.testing.assert_array_equal(actor.eyes[0].response, previous_eye.response)
                assert actor.eyes[0].time == previous_eye.time
            previous_eye = actor.eyes[0]
            action, packed = runner.step(actor)
            if tick == 2:
                assert not packed["ray_valid"][0].any()
            assert np.isfinite(action).all() and packed["organs"].shape == (60,)
            obs, _, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
        assert info["time"] > .8  # Actual physics, not a fabricated sensor fixture.
    finally:
        env.close()
