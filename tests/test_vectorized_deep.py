"""Deep agents with several environments, losses, and save/load of the new format."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import relepy  # noqa: E402
from relepy import A2C, DQN, PPO, REINFORCE, SAC, TD3, PPOConfig  # noqa: E402
from relepy.core import losses  # noqa: E402
from relepy.core.buffers import ReplayBuffer, RolloutBuffer  # noqa: E402

SMALL = dict(hidden_sizes=(16, 16), device="cpu", verbose=0, seed=0)

VEC_AGENTS = [
    (DQN, "CartPole-v1", dict(learning_starts=50, batch_size=16, buffer_size=1_000)),
    (REINFORCE, "CartPole-v1", dict(episodes_per_update=2)),
    (A2C, "CartPole-v1", dict(n_steps=8)),
    (PPO, "CartPole-v1", dict(n_steps=32, batch_size=32, n_epochs=2)),
    (PPO, "Pendulum-v1", dict(n_steps=32, batch_size=32, n_epochs=2)),
    (TD3, "Pendulum-v1", dict(learning_starts=50, batch_size=32, buffer_size=1_000)),
    (SAC, "Pendulum-v1", dict(learning_starts=50, batch_size=32, buffer_size=1_000)),
]


@pytest.mark.parametrize("cls,env_id,kwargs", VEC_AGENTS,
                         ids=[f"{c.__name__}-{e}" for c, e, _ in VEC_AGENTS])
def test_vectorized_training_and_roundtrip(cls, env_id, kwargs, tmp_path):
    agent = cls(env_id, n_envs=4, **kwargs, **SMALL)
    agent.fit(400)
    assert agent.num_timesteps >= 400 and agent.num_timesteps % 4 == 0
    assert agent.n_envs == 4 and len(agent.episode_rewards) >= 0

    path = tmp_path / "agent.relepy"
    agent.save(path)
    loaded = relepy.load(path, env_id, device="cpu", verbose=0)  # n_envs comes from the file
    assert loaded.n_envs == 4 and loaded.num_timesteps == agent.num_timesteps
    obs, _ = agent.env.reset(seed=3)
    np.testing.assert_allclose(loaded.predict(obs), agent.predict(obs), atol=1e-6)
    single = cls.load(path, env_id, device="cpu", verbose=0, n_envs=1)
    assert single.n_envs == 1


def test_ppo_batch_size_counts_all_envs():
    PPOConfig(n_steps=128, n_envs=8, batch_size=256)  # 1024 samples per update: valid
    with pytest.raises(ValueError, match="n_steps \\* n_envs"):
        PPOConfig(n_steps=16, n_envs=2, batch_size=64)


def test_ppo_target_kl_stops_before_the_step():
    agent = PPO("CartPole-v1", n_steps=64, batch_size=32, n_epochs=4, target_kl=1e-12, **SMALL)
    before = [p.detach().clone() for p in agent.policy.parameters()]
    agent.fit(64, log_interval=1)
    row = agent.logger.history[-1]
    # first mini-batch has KL 0 (identical policy) -> one step, then the second one stops
    assert row["train/stopped_early"] == 1.0
    changed = any(not torch.equal(a, b) for a, b in zip(before, agent.policy.parameters()))
    assert changed


def test_dqn_target_update_schedule_with_several_envs():
    agent = DQN("CartPole-v1", n_envs=4, target_update_interval=8, learning_starts=10_000,
                **SMALL)
    calls = []
    agent._update_target = lambda: calls.append(agent.num_timesteps)
    agent.fit(40)
    assert calls == [8, 16, 24, 32, 40]


def test_rollout_buffer_gae_per_env():
    buf = RolloutBuffer(n_steps=3, obs_dim=1, n_envs=2)
    for t in range(3):
        buf.add(np.zeros((2, 1)), np.zeros(2), [1.0, 2.0], [t == 1, False], [0.0, 0.0],
                [0.0, 0.0])
    buf.compute_returns_and_advantages(np.array([0.0, 10.0]), gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(buf.returns[:, 0], [2.0, 1.0, 1.0])  # env 0: episode ends at t=1
    np.testing.assert_allclose(buf.returns[:, 1], [16.0, 14.0, 12.0])  # env 1: bootstraps 10
    assert sum(b.observations.shape[0] for b in buf.get(4)) == 6


def test_replay_buffer_extend_wraps():
    buf = ReplayBuffer(capacity=4, obs_dim=1, action_shape=(1,))
    buf.extend(np.arange(3.0)[:, None], np.arange(3), np.ones(3), np.arange(3.0)[:, None],
               np.zeros(3))
    buf.extend(np.arange(3.0, 6.0)[:, None], np.arange(3), np.ones(3),
               np.arange(3.0)[:, None], np.zeros(3))
    assert len(buf) == 4 and buf.pos == 2
    assert sorted(buf.observations.ravel().tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_td_target_and_normalize():
    r = torch.tensor([[1.0], [1.0]])
    d = torch.tensor([[0.0], [1.0]])
    v = torch.tensor([[10.0], [10.0]])
    assert losses.td_target(r, d, v, 0.5).flatten().tolist() == [6.0, 1.0]
    x = losses.normalize(torch.tensor([1.0, 2.0, 3.0]))
    assert abs(float(x.mean())) < 1e-6
    single = torch.tensor([5.0])
    assert losses.normalize(single) is single


def test_loaded_optimizer_state_continues_training(tmp_path):
    agent = DQN("CartPole-v1", learning_starts=20, batch_size=16, buffer_size=500, **SMALL)
    agent.fit(100)
    path = tmp_path / "dqn.relepy"
    agent.save(path)
    loaded = DQN.load(path, "CartPole-v1", device="cpu", verbose=0)
    state_a = agent.optimizer.state_dict()["state"]
    state_b = loaded.optimizer.state_dict()["state"]
    assert state_a.keys() == state_b.keys()
    for k in state_a:
        torch.testing.assert_close(state_a[k]["exp_avg"], state_b[k]["exp_avg"])
    loaded.fit(50)  # keeps training without errors
