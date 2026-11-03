import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.graph import demo_graph
from flybrain_drive.learning import advantages
from flybrain_drive.policy import RecurrentPolicy, log_probability, tensor_observation


def test_timeout_bootstraps_without_crossing_reset():
    adv = advantages(
        np.array([1.0, 100.0]),
        np.zeros(2),
        np.array([10.0, 0.0]),
        np.array([False, True]),
        np.array([True, True]),
        gamma=0.9,
        lam=1,
    )
    np.testing.assert_allclose(adv, [10, 100])
    adv_terminated = advantages(
        np.array([1.0]), np.zeros(1), np.array([10.0]), np.array([True]), np.array([True])
    )
    np.testing.assert_allclose(adv_terminated, [1])


@pytest.mark.parametrize("kind", ["gru", "graph"])
def test_state_persistence_reset_detach_and_sequence_replay(kind):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    env = FlyBrainEnv()
    try:
        obs, _ = env.reset(seed=1)
        policy = RecurrentPolicy(kind, demo_graph() if kind == "graph" else None)
        tensors = tensor_observation(obs)
        zero = policy.initial_state()
        mean0, _, state1 = policy(tensors, zero)
        mean1, _, state2 = policy(tensors, state1.detach())
        assert not torch.equal(state1, state2)
        reset_mean, _, reset_state = policy(tensors, state2, torch.ones(1))
        torch.testing.assert_close(reset_mean, mean0)
        torch.testing.assert_close(reset_state, state1)
        sequence_obs = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in tensors.items()
        }
        replay_mean, _ = policy.sequence(sequence_obs, torch.tensor([True, False]))
        torch.testing.assert_close(replay_mean, torch.cat((mean0, mean1)))
    finally:
        env.close()


def test_camera_mode_oracle_channels_are_masked_and_encoder_ignores_them():
    # No renderer needed to test the policy-side information firewall.
    env = FlyBrainEnv()
    try:
        obs, _ = env.reset()
        obs["body_valid"][-6:] = 0
        policy = RecurrentPolicy()
        first = policy.encoder(tensor_observation(obs))
        obs["body"][-6:] = 99
        second = policy.encoder(tensor_observation(obs))
        torch.testing.assert_close(first, second)
        lp = log_probability(torch.zeros(1, 2), torch.zeros(2), torch.tensor([[50.0, -50.0]]))
        assert torch.isfinite(lp).all()
    finally:
        env.close()
