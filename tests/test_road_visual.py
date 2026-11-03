import numpy as np
import pytest

pytest.importorskip("mujoco")

from flybrain_drive.env import FlyBrainEnv


def test_continuous_road_changes_pixels_without_changing_physics_or_oracle_mask():
    settings = {"mode": "V", "road": {"flattened": True}, "max_steps": 8,
                "camera": {"source_size": 96, "fovy": 120, "forward": .95,
                           "pitch_degrees": -15, "near_m": .03, "far_m": 250}}
    legacy = FlyBrainEnv(**settings)
    visible = FlyBrainEnv(**settings, road_visual="lane_surface")
    try:
        obs_old, _ = legacy.reset(seed=17)
        obs_new, _ = visible.reset(seed=17)
        assert legacy.model.ngeom == visible.model.ngeom
        assert len(visible._road_visuals) == 600
        assert visible.configuration()["road_visual"] == "lane_surface"
        assert np.mean(np.abs(legacy.camera_source_images.astype(float)
                              - visible.camera_source_images)) > 3
        assert not np.array_equal(obs_old["images"], obs_new["images"])
        source = visible.camera_source_images.astype(float)
        assert np.sum((source[..., 1] > source[..., 0] * 1.05) & (source[..., 1] > 30)) > 10
        assert not obs_new["body_valid"][-6:].any()
        for _ in range(5):
            old, _, _, _, old_info = legacy.step([-.25, .03])
            new, _, _, _, new_info = visible.step([-.25, .03])
            np.testing.assert_array_equal(legacy.data.qpos, visible.data.qpos)
            np.testing.assert_array_equal(old["body"], new["body"])
            assert old_info["truth"] == new_info["truth"]
    finally:
        legacy.close()
        visible.close()
    with pytest.raises(ValueError, match="road_visual"):
        FlyBrainEnv(road_visual="unknown")
