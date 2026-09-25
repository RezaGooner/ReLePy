import numpy as np
import pytest

from relepy import ExpectedSARSA, GridWorldEnv, QLearning, SARSA
from relepy.core.callbacks import EvalCallback, StopOnRewardThreshold


@pytest.mark.parametrize("cls", [QLearning, SARSA, ExpectedSARSA])
def test_learns_gridworld(cls):
    env = GridWorldEnv(size=4)
    agent = cls(env, alpha=0.3, epsilon_decay_steps=3_000, seed=0, verbose=0)
    agent.fit(10_000)
    mean, _ = agent.evaluate(GridWorldEnv(size=4), n_episodes=5)
    assert mean > 0.8  # optimal: 6 moves -> 1.0 - 5 * 0.01 = 0.95


def test_save_load_roundtrip(tmp_path):
    agent = QLearning(GridWorldEnv(), seed=1, verbose=0).fit(2_000)
    path = tmp_path / "q.relepy"
    agent.save(path)
    loaded = QLearning.load(path, GridWorldEnv(), verbose=0)
    np.testing.assert_allclose(loaded.q_table, agent.q_table)
    assert loaded.num_timesteps == agent.num_timesteps
    with pytest.raises(ValueError):
        SARSA.load(path, GridWorldEnv(), verbose=0)


def test_seed_reproducibility():
    a = QLearning(GridWorldEnv(), seed=7, verbose=0).fit(1_000)
    b = QLearning(GridWorldEnv(), seed=7, verbose=0).fit(1_000)
    np.testing.assert_allclose(a.q_table, b.q_table)


def test_rejects_wrong_spaces():
    gym = pytest.importorskip("gymnasium")
    with pytest.raises(TypeError):
        QLearning(gym.make("CartPole-v1"), verbose=0)


def test_unknown_hyperparameter_and_config_type():
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        QLearning(GridWorldEnv(), learning_rate=0.1, verbose=0)
    from relepy.core.config import BaseConfig

    with pytest.raises(TypeError):
        QLearning(GridWorldEnv(), BaseConfig(), verbose=0)


def test_callbacks_stop_training():
    agent = QLearning(GridWorldEnv(), alpha=0.5, seed=0, verbose=0)
    cb = StopOnRewardThreshold(reward_threshold=0.5, min_episodes=10)
    agent.fit(50_000, callbacks=cb)
    assert agent.num_timesteps < 50_000

    agent2 = QLearning(GridWorldEnv(), alpha=0.5, seed=0, verbose=0)
    ev = EvalCallback(GridWorldEnv(), eval_freq=500, n_eval_episodes=2)
    agent2.fit(1_500, callbacks=ev)
    assert len(ev.history) == 3
