import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from flybrain_drive.control import IntentController, VehicleConfig
from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run
from flybrain_drive.imitation import sequence_chunks, teacher_action, teacher_warmstart
from flybrain_drive.learning import restore_policy
from flybrain_drive.parallel_learning import train_parallel
from flybrain_drive.policy import log_probability, policy_from_config, tensor_observation
from flybrain_drive.vision import CameraConfig, MotionFilters, RaySampler


def test_motor_allocation_limits_rates_and_positive_left_yaw():
    vehicle = VehicleConfig()
    controller = IntentController(vehicle, "residual")
    previous = np.zeros(2)
    for _ in range(40):
        command, info = controller.command([-.5, 0, 1], 2.0, 0.0, np.ones(4) * 9, 0.01)
        torque = command[2:4]
        assert np.all(torque >= 0) and np.all(torque <= vehicle.max_torque)
        assert np.max(np.abs(torque - previous)) <= vehicle.max_torque_rate * .01 + 1e-8
        previous = torque.copy()
    # At zero common drive torque there is no feasible positive-drive yaw authority.
    np.testing.assert_allclose(previous, 0)
    controller.reset()
    for _ in range(40):
        command, info = controller.command([0, 0, 1], 1.0, 0.0, np.ones(4) * 5, 0.01)
    assert command[3] > command[2]
    np.testing.assert_allclose(info["residual_yaw_moment_requested"], 80)
    np.testing.assert_allclose(info["yaw_moment_from_commanded_torques"], 80)
    off = IntentController(vehicle, "off")
    fixed = IntentController(vehicle, "fixed")
    assert off.command([0, .5, 1], 1, 0, np.ones(4), .1)[1]["yaw_moment_requested"] == 0
    assert fixed.command([0, .5, 1], 1, 0, np.ones(4), .1)[1]["residual_yaw_moment_requested"] == 0


def test_positive_residual_moment_turns_the_physical_vehicle_left():
    results = []
    for residual in (0., .5):
        env = FlyBrainEnv(actuation_mode="residual", vehicle={"max_speed": 4}, road={
            "turn_amplitude": 0, "hill_amplitude": 0, "bank_degrees": 0,
            "bump_height": 0, "grade": 0,
        })
        try:
            env.reset(seed=1)
            for _ in range(200):
                env.step(np.array([0, 0, residual], np.float32))
            results.append((env.data.qpos[1], float(env.data.sensor("gyro").data[2])))
        finally:
            env.close()
    # Verify the sign, not an uncalibrated yaw gain. The current cylinder tire
    # contacts strongly resist yaw: this probe only gives millimetres of offset.
    assert results[1][0] > results[0][0] + 1e-4
    assert results[1][1] > results[0][1] + 1e-4


def test_ray_directions_masks_and_identity_camera():
    ordinary = RaySampler(CameraConfig(), 32)
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    np.testing.assert_array_equal(ordinary.sample(image), image)
    sampler = RaySampler(CameraConfig(
        fovy=120, source_size=96, sampling="angular_hex", angular_width=150
    ), 32)
    np.testing.assert_allclose(np.linalg.norm(sampler.rays, axis=-1), 1, atol=1e-6)
    assert np.all(sampler.rays[..., 2] < 0)
    assert np.any(sampler.valid == 0) and np.any(sampler.valid == 1)
    assert np.all(sampler.sample(np.full((96, 96, 3), 255, np.uint8))[sampler.valid == 0] == 0)


def test_explicit_metric_clipping_restores_nearby_road_without_changing_physics():
    positions, visible = [], []
    camera = {"yaw_degrees": 40, "pitch_degrees": -15, "fovy": 120,
              "forward": .95, "source_size": 96, "sampling": "angular_hex"}
    for clipping in ({}, {"near_m": .03, "far_m": 250}):
        env = FlyBrainEnv(mode="V", visual_schema="retina_v1", camera={**camera, **clipping})
        try:
            env.reset(seed=1)
            positions.append(env.data.qpos.copy())
            visible.append(float((env.camera_source_images.sum(-1) > 0).mean()))
            if clipping:
                np.testing.assert_allclose(env.configuration()["render_clip_m"]["near"], .03)
                assert env.camera_source_images[:, -1, 48].sum() > 0
        finally:
            env.close()
    np.testing.assert_array_equal(*positions)
    assert visible[1] > visible[0] + .25
    with pytest.raises(ValueError, match="near_m"):
        CameraConfig(near_m=.03)


def test_motion_filters_direction_no_wrap_actual_clock_and_reset():
    filters = MotionFilters(np.ones((2, 16, 16), np.float32))
    image = np.zeros((2, 16, 16, 3), np.uint8)
    initial, valid = filters.update(image, .8)
    assert not initial.any() and not valid.any()
    image[:, :, 5:7] = 255
    filters.update(image, .84)
    image[:] = 0
    image[:, :, 6:8] = 255
    features, valid = filters.update(image, .88)
    assert features[..., 2].sum() > features[..., 3].sum()  # right vs left
    assert not valid[:, :, -1, 2].any()
    with pytest.raises(ValueError, match="timestamps"):
        filters.update(image, .88)
    filters.reset()
    features, valid = filters.update(image, 0)
    assert not features.any() and not valid.any()


@pytest.mark.parametrize("mode", ["off", "fixed", "residual"])
def test_new_sensor_policy_schema_masks_and_checkpoint_dimensions(mode):
    config = {"motion_input": True, "environment": {
        "mode": "V", "visual_schema": "retina_v1", "actuation_mode": mode,
        "camera_hz": 25, "camera": {"sampling": "angular_hex", "source_size": 96,
                                    "fovy": 120, "pitch_degrees": -15, "yaw_degrees": 40},
    }}
    env = FlyBrainEnv(**config["environment"])
    try:
        observation, _ = env.reset(seed=1)
        assert env.observation_space.contains(observation)
        assert env.action_space.shape == (3,)
        np.testing.assert_array_equal(observation["body_valid"][-6:], 0)
        policy = policy_from_config(config)
        with torch.no_grad():
            mean, value, state = policy(tensor_observation(observation), policy.initial_state())
        assert mean.shape == (1, 3) and torch.isfinite(value).all()
        assert policy.action_mask[2] == (mode == "residual")
        after, _, _, _, _ = env.step(np.array([0, 0, 1], np.float32))
        assert after["previous_yaw_action"][0] == (mode == "residual")
        assert after["camera_frame_id"][0] == observation["camera_frame_id"][0]
        assert not after["motion_valid"].any()
        assert env.observation_space.contains(after)
        # Invalid retinal samples cannot change either RGB or motion embedding.
        obs = tensor_observation(after)
        obs["retina_valid"].zero_()
        first = policy.encoder(obs)
        obs["images"].fill_(255)
        obs["motion"].fill_(1)
        torch.testing.assert_close(first, policy.encoder(obs))
    finally:
        env.close()


def test_disabled_action_does_not_enter_ppo_likelihood():
    mean = torch.zeros(4, 3)
    log_std = torch.zeros(3)
    mask = torch.tensor([1., 1., 0.])
    latent = torch.zeros(4, 3)
    first = log_probability(mean, log_std, latent, mask)
    latent[:, 2] = 3
    torch.testing.assert_close(first, log_probability(mean, log_std, latent, mask))


def test_teacher_uses_route_labels_while_sensor_actor_keeps_oracle_masked():
    env = FlyBrainEnv(mode="V", actuation_mode="residual", visual_schema="retina_v1")
    try:
        observation, info = env.reset(seed=1)
        action = teacher_action(env, info)
        assert action.shape == (3,) and action[2] == 0
        assert np.all(observation["body"][-6:] == 0)
        assert np.all(observation["body_valid"][-6:] == 0)
    finally:
        env.close()


def test_parallel_bc_to_ppo_roundtrip_keeps_teacher_truth_out_of_actor_dataset(tmp_path):
    config = {
        "device": "cpu", "num_envs": 2, "torch_threads": 1, "seed": 22,
        "horizon": 4, "updates": 2, "epochs": 1, "collector_restart": False,
        "motion_input": True, "initial_log_std": [-1.7, -2.5, -1.7],
        "environment": {"mode": "V", "actuation_mode": "residual",
                        "visual_schema": "retina_v1", "camera_hz": 25},
        "behavior_cloning": {"epochs": 2, "chunk_length": 4,
                             "episodes": [{"start_x": 3, "steps": 8}]},
    }
    run = train_parallel(config, tmp_path / "train")
    initial, _ = restore_policy(run / "initial-checkpoint.pt")
    final, restored = restore_policy(run / "checkpoint.pt")
    assert final.actor.out_features == 3 and final.encoder.motion_input
    assert restored["behavior_cloning"]["epochs"] == 2
    assert (run / "random-checkpoint.pt").exists() and (run / "behavior-cloning.json").exists()
    assert any(not torch.equal(a, b) for a, b in zip(
        initial.parameters(), final.parameters(), strict=True
    ))
    with np.load(run / "teacher-data" / "episode-00.npz") as data:
        np.testing.assert_array_equal(data["obs_body"][:, -6:], 0)
        np.testing.assert_array_equal(data["obs_body_valid"][:, -6:], 0)
        assert data["action"].shape == (8, 3)
        assert not any("qpos" in key or "road_frame" in key for key in data.files)
        np.testing.assert_allclose(np.diff(data["observation_time"]), .01)


def test_canonical_teacher_gives_identical_bc_initial_weights_across_tv_arms(tmp_path):
    weights = []
    for mode in ("off", "fixed", "residual"):
        torch.manual_seed(31)
        config = {
            "environment": {"mode": "V", "actuation_mode": mode,
                            "visual_schema": "retina_v1"},
            "behavior_cloning": {"teacher_actuation_mode": "fixed", "train_yaw_action": False,
                                 "epochs": 2, "chunk_length": 4,
                                 "batching": "parallel_episodes",
                                 "episodes": [{"start_x": 3, "steps": 8, "lateral_offset": .6,
                                               "heading_offset_degrees": 4}],
                                 "dagger_rounds": [{"teacher_beta": .5, "epochs": 1,
                                                    "episodes": [{"start_x": 3, "steps": 4}]}]},
        }
        policy = policy_from_config(config)
        run = begin_run(tmp_path / mode, config)
        teacher_warmstart(policy, config, run, torch.device("cpu"))
        weights.append({key: value.clone() for key, value in policy.state_dict().items()})
    for candidate in weights[1:]:
        for key in weights[0]:
            torch.testing.assert_close(weights[0][key], candidate[key], rtol=0, atol=0)


def test_episode_batch_padding_masks_only_real_samples_and_preserves_boundaries():
    episodes = [{"observation": {"x": np.arange(length, dtype=np.float32)[:, None]},
                 "action": np.ones((length, 3), np.float32)} for length in (5, 2)]
    chunks = list(sequence_chunks(episodes, 3, torch.device("cpu")))
    assert sum(int(valid.sum()) for _, _, _, valid in chunks) == 7
    torch.testing.assert_close(chunks[0][2][0], torch.ones(2))  # declared start
    torch.testing.assert_close(chunks[1][2][:2, 0], torch.zeros(2))  # continuous long episode
    assert torch.all(chunks[1][2][:, 1] == 1)  # no numerical carry through padding
    assert not chunks[1][3][:, 1].any()


def test_dagger_collects_expert_labels_at_student_states_without_pose_inputs(tmp_path):
    torch.manual_seed(23)
    config = {
        "environment": {"mode": "V", "actuation_mode": "residual",
                        "visual_schema": "retina_v1"},
        "behavior_cloning": {
            "teacher_actuation_mode": "fixed", "train_yaw_action": False,
            "batching": "parallel_episodes", "epochs": 1, "chunk_length": 4,
            "episodes": [{"start_x": 3, "steps": 5}, {"start_x": 35, "steps": 3}],
            "dagger_rounds": [{"teacher_beta": 0, "epochs": 1, "data_seed": 9911,
                               "episodes": [{"start_x": 3, "steps": 8}]}],
        },
    }
    run = begin_run(tmp_path / "dagger", config)
    summary = teacher_warmstart(policy_from_config(config), config, run, torch.device("cpu"))
    assert summary["epochs"][0]["supervised_samples"] == 8
    assert summary["epochs"][1]["supervised_samples"] == 16
    with np.load(run / "dagger-data-0" / "episode-00.npz") as data:
        assert not data["teacher_selected"].any()
        assert np.max(np.abs(data["executed_action"] - data["action"])) > .01
        np.testing.assert_array_equal(data["obs_body"][:, -6:], 0)
        np.testing.assert_array_equal(data["executed_action"][:, 2], 0)
        assert not any("qpos" in key or "road_frame" in key for key in data.files)
