import numpy as np
import pytest

pytest.importorskip("mujoco")

from flybrain_drive.control import IntentController
from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.road import RoadConfig, SyntheticRoad


def test_true_3d_progress_and_frames():
    road = SyntheticRoad(
        RoadConfig(grade=0.1, hill_amplitude=0, bank_degrees=5, turn_amplitude=0, bump_height=0)
    )
    assert road.s[-1] > road.config.length
    frame = road.frame(25)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-6)
    assert np.linalg.det(frame) > 0.999
    assert frame[2, 0] > 0
    assert frame[2, 1] > 0
    p = road.xyz[500]
    s, ey, projected, _ = road.project(p + frame[:, 1] * 0.5)
    assert abs(ey - 0.5) < 0.02
    assert s > 20
    assert abs(projected[2] - road.height(projected[0], projected[1])) < 0.01


@pytest.mark.parametrize("grade,expected", [(0.08, -1), (-0.08, 1)])
def test_gravity_drives_unpowered_vehicle_downhill(grade, expected):
    env = FlyBrainEnv(
        road={
            "grade": grade,
            "hill_amplitude": 0,
            "bank_degrees": 0,
            "turn_amplitude": 0,
            "bump_height": 0,
        }
    )
    try:
        env.reset(seed=1)
        x = env.data.qpos[0]
        env.data.ctrl[:] = 0
        for _ in range(500):
            env.mj.mj_step(env.model, env.data)
        assert expected * (env.data.qpos[0] - x) > 0.01
        assert env.data.ncon > 0
        assert np.all(np.isfinite(env.data.qpos))
    finally:
        env.close()


def test_controller_left_turn_sign_and_no_oracle_dependency():
    controller = IntentController()
    command, _ = controller.command(np.array([0.0, 0.5]), 2, 0, np.ones(4) * 10, 0.01)
    assert command[0] > 0 and command[1] > 0
    assert command[3] > command[2]
    stopped, info = controller.command(
        np.array([1.0, 1.0]), 2, 1, np.ones(4) * 10, 0.01, valid=False
    )
    assert info["v_ref"] == 0
    assert np.all(stopped[2:4] == 0)
    assert np.all(stopped[4:] < 0)


def test_reset_reproducibility_and_timeout_contract():
    env = FlyBrainEnv(max_steps=2)
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first["body"], second["body"])
        assert env.observation_space.contains(first)
        env.step(np.array([-0.5, 0], dtype=np.float32))
        _, _, terminated, truncated, info = env.step(np.array([-0.5, 0], dtype=np.float32))
        assert not terminated and truncated and info["reason"] == "time_limit"
        with pytest.raises(RuntimeError):
            env.step(np.zeros(2))
        with pytest.raises(ValueError, match="M is planned"):
            FlyBrainEnv(mode="M")
    finally:
        env.close()


def test_real_camera_mode_masks_oracle_and_reuses_exposure():
    env = FlyBrainEnv(mode="V", camera_hz=30)
    try:
        obs, _ = env.reset(seed=1)
        assert np.all(obs["body"][-6:] == 0)
        assert np.all(obs["body_valid"][-6:] == 0)
        assert np.all(obs["camera_valid"] == 1)
        assert obs["images"].std() > 1
        assert not np.array_equal(obs["images"][0], obs["images"][1])
        following, _, _, _, _ = env.step(np.array([-0.75, 0], dtype=np.float32))
        np.testing.assert_array_equal(following["images"], obs["images"])
        np.testing.assert_allclose(following["camera_age"], 0.01, atol=1e-6)
        assert env.observation_space.contains(following)
    finally:
        env.close()
