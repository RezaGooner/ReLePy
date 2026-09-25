import numpy as np
import pytest

pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from relepy import A2C, DQN, PPO, REINFORCE, DoubleDQN, DuelingDQN  # noqa: E402

SMALL = dict(hidden_sizes=(16, 16), device="cpu", verbose=0, seed=0)

DISCRETE_AGENTS = [
    (DQN, dict(learning_starts=50, batch_size=16, buffer_size=1_000)),
    (DoubleDQN, dict(learning_starts=50, batch_size=16, buffer_size=1_000)),
    (DuelingDQN, dict(learning_starts=50, batch_size=16, buffer_size=1_000, tau=0.5,
                      target_update_interval=10)),
    (REINFORCE, dict(episodes_per_update=2)),
    (A2C, dict(n_steps=8)),
    (PPO, dict(n_steps=64, batch_size=32, n_epochs=2)),
]


@pytest.mark.parametrize("cls,kwargs", DISCRETE_AGENTS, ids=lambda x: getattr(x, "__name__", ""))
def test_smoke_train_predict_save_load(cls, kwargs, tmp_path):
    agent = cls("CartPole-v1", **kwargs, **SMALL)
    agent.fit(300, log_interval=1)
    assert agent.num_timesteps >= 300
    obs, _ = agent.env.reset()
    assert agent.action_space.contains(agent.predict(obs))
    mean, _ = agent.evaluate(n_episodes=2)
    assert np.isfinite(mean)

    path = tmp_path / "agent.relepy"
    agent.save(path)
    loaded = cls.load(path, "CartPole-v1", verbose=0, device="cpu")
    for _ in range(5):
        obs, _ = agent.env.reset()
        assert loaded.predict(obs) == agent.predict(obs)


@pytest.mark.parametrize("cls,kwargs", [
    (PPO, dict(n_steps=64, batch_size=32, n_epochs=2)),
    (A2C, dict(n_steps=8)),
    (REINFORCE, dict(episodes_per_update=1)),
], ids=lambda x: getattr(x, "__name__", ""))
def test_continuous_actions(cls, kwargs):
    agent = cls("Pendulum-v1", **kwargs, **SMALL)
    agent.fit(300)
    obs, _ = agent.env.reset()
    action = agent.predict(obs)
    assert agent.action_space.contains(action)


def test_dqn_rejects_continuous_actions():
    with pytest.raises(TypeError):
        DQN("Pendulum-v1", verbose=0)


def test_forced_config():
    assert DoubleDQN("CartPole-v1", verbose=0, device="cpu").config.double_dqn is True
    assert DuelingDQN("CartPole-v1", verbose=0, device="cpu").config.dueling is True


def test_discrete_observation_env_works_with_deep_agent():
    from relepy import GridWorldEnv

    agent = DQN(GridWorldEnv(), learning_starts=20, batch_size=8, **SMALL)
    agent.fit(100)
    assert agent.action_space.contains(agent.predict(0))


@pytest.mark.slow
def test_ppo_learns_cartpole():
    agent = PPO("CartPole-v1", n_steps=512, batch_size=64, hidden_sizes=(64, 64),
                device="cpu", verbose=0, seed=0)
    agent.fit(60_000)
    mean, _ = agent.evaluate(n_episodes=10)
    assert mean > 150


@pytest.mark.slow
def test_dqn_learns_cartpole():
    agent = DoubleDQN("CartPole-v1", learning_rate=1e-3, hidden_sizes=(64, 64), device="cpu",
                      verbose=0, seed=0, epsilon_decay_steps=8_000, target_update_interval=250)
    agent.fit(40_000)
    mean, _ = agent.evaluate(n_episodes=10)
    assert mean > 100
