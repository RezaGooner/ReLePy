import numpy as np
import pytest

pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from relepy import DDPG, SAC, TD3, DDPGConfig, TD3Config  # noqa: E402

SMALL = dict(
    learning_starts=50, batch_size=32, buffer_size=1_000, hidden_sizes=(16, 16),
    device="cpu", verbose=0, seed=0,
)


@pytest.mark.parametrize("cls", [DDPG, TD3, SAC])
def test_smoke_train_predict_save_load(cls, tmp_path):
    agent = cls("Pendulum-v1", **SMALL)
    agent.fit(200, log_interval=1)
    assert agent.num_timesteps >= 200
    obs, _ = agent.env.reset()
    action = agent.predict(obs)
    assert agent.action_space.contains(action)
    assert agent.action_space.contains(agent.predict(obs, deterministic=False))

    path = tmp_path / "agent.relepy"
    agent.save(path)
    loaded = cls.load(path, "Pendulum-v1", verbose=0, device="cpu")
    np.testing.assert_allclose(loaded.predict(obs), agent.predict(obs), atol=1e-6)
    mean, _ = loaded.evaluate(n_episodes=1)
    assert np.isfinite(mean)


@pytest.mark.parametrize("cls", [DDPG, TD3, SAC])
def test_rejects_discrete_actions(cls):
    with pytest.raises(TypeError, match="continuous"):
        cls("CartPole-v1", verbose=0)


def test_action_scaling_to_env_bounds():
    agent = DDPG("Pendulum-v1", **SMALL)  # Pendulum torque range is [-2, 2]
    scale = agent._to_env_action
    np.testing.assert_allclose(scale(np.array([-1.0], np.float32)), [-2.0])
    np.testing.assert_allclose(scale(np.array([0.0], np.float32)), [0.0], atol=1e-6)
    np.testing.assert_allclose(scale(np.array([1.0], np.float32)), [2.0])


def test_td3_is_ddpg_with_different_defaults():
    ddpg, td3 = DDPGConfig(), TD3Config()
    assert (ddpg.n_critics, ddpg.policy_delay, ddpg.target_policy_noise) == (1, 1, 0.0)
    assert (td3.n_critics, td3.policy_delay, td3.target_policy_noise) == (2, 2, 0.2)
    assert len(TD3("Pendulum-v1", **SMALL).critic.q_nets) == 2


def test_sac_fixed_temperature():
    agent = SAC("Pendulum-v1", auto_entropy=False, initial_alpha=0.2, **SMALL)
    agent.fit(150)
    assert agent.alpha == pytest.approx(0.2, rel=1e-4)


def test_off_policy_config_validation():
    with pytest.raises(ValueError):
        DDPGConfig(tau=0.0)
    with pytest.raises(ValueError):
        DDPGConfig(policy_delay=0)


@pytest.mark.slow
def test_sac_learns_pendulum():
    agent = SAC("Pendulum-v1", learning_starts=1_000, device="cpu", verbose=0, seed=0)
    agent.fit(15_000)
    mean, _ = agent.evaluate(n_episodes=5)
    assert mean > -700  # a random policy scores about -1200
