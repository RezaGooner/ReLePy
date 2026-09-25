"""Q-learning on the built-in GridWorld (NumPy only)."""

from relepy import GridWorldEnv, QLearning

agent = QLearning(GridWorldEnv(size=5), alpha=0.3, epsilon_decay_steps=5_000, seed=0)
agent.fit(15_000, log_interval=100)
mean, std = agent.evaluate(GridWorldEnv(size=5), n_episodes=10)
print(f"evaluation: {mean:.3f} +/- {std:.3f}")
print("Greedy action per state:\n", agent.q_table.argmax(axis=1).reshape(5, 5))
