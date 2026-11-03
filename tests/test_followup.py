import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evaluation import evaluate_suite
from flybrain_drive.graph import demo_graph
from flybrain_drive.parallel_learning import train_parallel
from flybrain_drive.policy import RecurrentPolicy
from scripts.run_campaign import merged, paired_comparisons


@pytest.mark.parametrize("kind", ["gru", "graph"])
def test_sequence_replays_nonzero_boundary_state(kind):
    torch.set_num_threads(1)
    policy = RecurrentPolicy(kind, demo_graph() if kind == "graph" else None)
    obs = {
        "images": torch.zeros(3, 2, 2, 32, 32, 3, dtype=torch.uint8),
        "body": torch.randn(3, 2, 38),
        "body_valid": torch.ones(3, 2, 38),
        "camera_valid": torch.zeros(3, 2, 2),
        "camera_age": torch.zeros(3, 2, 2),
    }
    initial = torch.randn(2, policy.state_size)
    resets = torch.tensor([[False, False], [True, False], [False, False]])
    means, values, final = policy.sequence_batch(obs, resets, initial, return_state=True)
    state = initial.clone()
    for tick in range(3):
        mean, value, state = policy(
            {key: val[tick] for key, val in obs.items()}, state, resets[tick]
        )
        torch.testing.assert_close(means[tick], mean)
        torch.testing.assert_close(values[tick], value)
    torch.testing.assert_close(final, state)


def test_continuous_rollout_survives_optimizer_boundary_and_evaluation_has_own_budget(tmp_path):
    config = {
        "device": "cpu",
        "policy": "gru",
        "seed": 3001,
        "num_envs": 2,
        "horizon": 5,
        "updates": 2,
        "epochs": 2,
        "checkpoint_interval": 1,
        "collector_restart": False,
        "target_kl": 0.03,
        "torch_threads": 1,
        "environment": {"mode": "O", "max_steps": 100},
    }
    run = train_parallel(config, tmp_path / "train")
    rows = json.loads((run / "training.json").read_text())["updates"]
    assert rows[0]["episode_steps_at_end"] == [5, 5]
    assert rows[1]["episode_steps_at_start"] == [5, 5]
    assert rows[1]["episode_steps_at_end"] == [10, 10]
    assert min(rows[1]["hidden_norm_at_start"]) > 0
    assert rows[1]["collector_end_reason"] == "continued_next_update"
    for row in rows:
        assert np.isfinite(row["approx_kl"]) and 0 <= row["clip_fraction"] <= 1
    evaluation = evaluate_suite(
        run / "initial-checkpoint.pt",
        tmp_path / "eval",
        {
            "steps": 4,
            "cases": [{"name": "offset", "start_x": 35, "road": {"grade": 0.06}}],
        },
    )
    case = evaluation["cases"][0]
    assert case["steps"] == 4 and case["reason"] == "time_limit"
    assert case["initial_progress_3d"] > 30
    assert case["environment"]["road"]["grade"] == 0.06
    assert case["environment"]["max_steps"] == 4


def test_no_proprioception_masks_measurements_retains_efference_copy_and_marks_invalid():
    env = FlyBrainEnv(mode="O", sensor_profile="no_proprioception")
    try:
        obs, _ = env.reset(seed=1)
        indices = np.r_[0:20, 24:32]
        np.testing.assert_array_equal(obs["body_valid"][indices], 0)
        np.testing.assert_array_equal(obs["body"][indices], 0)
        np.testing.assert_array_equal(obs["body_valid"][20:24], 1)
        obs, _, _, _, _ = env.step(np.array([0.1, 0.2]))
        assert np.any(obs["body"][20:24] != 0)
        np.testing.assert_array_equal(obs["body"][indices], 0)
        assert env.configuration()["sensor_profile"] == "no_proprioception"
    finally:
        env.close()


def test_nested_ablation_preserves_road_and_comparisons_pair_seed_case():
    base = {"environment": {"mode": "V", "road": {"grade": 0.02}}}
    config = merged(base, {"environment": {"sensor_profile": "no_proprioception"}})
    assert config["environment"]["mode"] == "V"
    assert config["environment"]["road"]["grade"] == 0.02
    assert "sensor_profile" not in base["environment"]
    case = {
        "case": "nominal",
        "progress_delta": 10,
        "completion": False,
        "failure": True,
        "lateral_error_rms": 1,
    }
    rows = [
        {"arm": "a", "seed": 3001, "status": "completed", "final": {"cases": [case]}},
        {
            "arm": "b",
            "seed": 3001,
            "status": "completed",
            "final": {
                "cases": [
                    {
                        **case,
                        "progress_delta": 12,
                        "failure": False,
                    }
                ]
            },
        },
        {"arm": "b", "seed": 3002, "status": "completed", "final": {"cases": [case]}},
    ]
    pairs = paired_comparisons(rows, ["a", "b"])
    assert len(pairs) == 1
    assert pairs[0]["progress_delta_difference"] == 2 and pairs[0]["failure_difference"] == -1
