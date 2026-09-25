"""PPO with 8 parallel CartPole environments (n_envs), which collects a rollout of
``n_steps * n_envs`` samples per update instead of a single environment's ``n_steps``."""

from relepy import PPO

agent = PPO(
    "CartPole-v1",          # env id (or a factory: lambda: gym.make("CartPole-v1"))
    n_envs=8,
    n_steps=128,            # 128 * 8 = 1024 samples collected per update
    batch_size=256,
    n_epochs=10,
    seed=0,
)
agent.fit(100_000, log_interval=5)
print(agent.evaluate(n_episodes=10))
