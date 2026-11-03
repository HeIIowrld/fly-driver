import json
import os
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from flybrain_drive.graph import demo_graph
from flybrain_drive.parallel_learning import train_parallel
from flybrain_drive.policy import RecurrentPolicy
from flybrain_drive.vector_env import RandomStartEnv


@pytest.mark.parametrize("kind", ["gru", "graph"])
def test_batched_sequence_matches_independent_rows_and_resets(kind):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    policy = RecurrentPolicy(kind, demo_graph() if kind == "graph" else None)
    length, count = 4, 2
    observation = {
        "images": torch.randint(0, 256, (length, count, 2, 32, 32, 3), dtype=torch.uint8),
        "body": torch.randn(length, count, 38),
        "body_valid": torch.ones(length, count, 38),
        "camera_valid": torch.ones(length, count, 2),
        "camera_age": torch.zeros(length, count, 2),
    }
    resets = torch.tensor([[True, True], [False, False], [True, False], [False, True]])
    means, values = policy.sequence_batch(observation, resets)
    for worker in range(count):
        row_obs = {key: value[:, worker] for key, value in observation.items()}
        row_means, row_values = policy.sequence(row_obs, resets[:, worker])
        torch.testing.assert_close(means[:, worker], row_means)
        torch.testing.assert_close(values[:, worker], row_values)


def test_partial_reset_keeps_other_workers_observation_and_clock():
    env = AsyncVectorEnv(
        [
            partial(RandomStartEnv, mode="O", max_steps=1),
            partial(RandomStartEnv, mode="O", max_steps=3),
        ],
        context="spawn",
        shared_memory=True,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    try:
        env.reset(seed=[100, 101])
        obs, _, terminated, truncated, _ = env.step(np.zeros((2, 2), dtype=np.float32))
        assert not terminated.any()
        np.testing.assert_array_equal(truncated, [True, False])
        preserved = obs["body"][1].copy()
        next_obs, _ = env.reset(seed=[102, None], options={"reset_mask": truncated.copy()})
        np.testing.assert_array_equal(next_obs["body"][1], preserved)
        _, _, _, next_truncated, _ = env.step(np.zeros((2, 2), dtype=np.float32))
        assert not next_truncated[1]
    finally:
        env.close(terminate=True)


def test_process_parallel_ppo_updates_and_records_worker_evidence(tmp_path):
    config = {
        "device": "cpu",
        "policy": "gru",
        "seed": 3001,
        "num_envs": 2,
        "horizon": 5,
        "updates": 2,
        "epochs": 1,
        "checkpoint_interval": 1,
        "torch_threads": 1,
        "start_x_range": [20, 60],
        "environment": {"mode": "O", "max_steps": 3},
    }
    train_parallel(config, tmp_path / "parallel")
    device = json.loads((tmp_path / "parallel/device.json").read_text())
    assert len(set(device["worker_pids"])) == 2
    assert os.getpid() not in device["worker_pids"]
    result = json.loads((tmp_path / "parallel/training.json").read_text())
    assert result["interactions"] == 20 and result["parameter_change_l2"] > 0
    assert all(row["transitions_per_second"] > 0 for row in result["updates"])
    assert all(len(row["episodes"]) == 2 for row in result["updates"])
