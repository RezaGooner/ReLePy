"""SAC (or swap in TD3 / DDPG) on a continuous-control task."""

from relepy import SAC, SACConfig

config = SACConfig(learning_rate=3e-4, batch_size=256, learning_starts=1_000, tau=0.005)
agent = SAC("Pendulum-v1", config, seed=0)
agent.fit(15_000, log_interval=10)
print(agent.evaluate(n_episodes=10))
