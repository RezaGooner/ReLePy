import numpy as np
import pytest

torch = pytest.importorskip("torch")

from relepy.core.buffers import ReplayBuffer, RolloutBuffer  # noqa: E402


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(capacity=5, obs_dim=2, rng=np.random.default_rng(0))
    for i in range(8):
        obs = np.array([i, i], dtype=np.float32)
        buf.add(obs, i % 2, float(i), obs + 1, i == 7)
    assert len(buf) == 5
    batch = buf.sample(4)
    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4, 1) and batch.actions.dtype == torch.int64
    assert batch.rewards.shape == (4, 1) and batch.dones.shape == (4, 1)
    assert batch.observations.min() >= 3  # oldest 3 transitions were overwritten


def test_gae_matches_reward_to_go_when_lambda_is_one():
    buf = RolloutBuffer(n_steps=4, obs_dim=1)
    rewards = [1.0, 1.0, 1.0, 1.0]
    dones = [False, False, True, False]
    for r, d in zip(rewards, dones):
        buf.add(np.zeros(1), 0, r, d, value=0.0, log_prob=0.0)
    buf.compute_returns_and_advantages(last_value=0.0, gamma=1.0, gae_lambda=1.0)
    # episode 1 = steps 0-2 (return 3, 2, 1); step 3 starts a new episode (return 1)
    np.testing.assert_allclose(buf.returns, [3.0, 2.0, 1.0, 1.0])


def test_gae_bootstraps_last_value():
    buf = RolloutBuffer(n_steps=2, obs_dim=1)
    for _ in range(2):
        buf.add(np.zeros(1), 0, 1.0, False, value=0.0, log_prob=0.0)
    buf.compute_returns_and_advantages(last_value=10.0, gamma=0.5, gae_lambda=1.0)
    np.testing.assert_allclose(buf.returns, [1.0 + 0.5 * (1.0 + 0.5 * 10.0), 1.0 + 0.5 * 10.0])
