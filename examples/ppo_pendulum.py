"""PPO with a continuous action space; hyperparameters passed as keyword arguments."""

from relepy import PPO

agent = PPO("Pendulum-v1", n_steps=2048, batch_size=64, n_epochs=10, entropy_coef=0.0, seed=0)
agent.fit(200_000, log_interval=5)
print(agent.evaluate(n_episodes=10))
