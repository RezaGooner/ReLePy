"""Double DQN on CartPole with periodic evaluation and a saved best model."""

from relepy import DoubleDQN, DQNConfig, EvalCallback

config = DQNConfig(
    learning_rate=1e-3,
    hidden_sizes=(64, 64),
    epsilon_decay_steps=8_000,
    target_update_interval=250,
)
agent = DoubleDQN("CartPole-v1", config, seed=42)
agent.fit(
    40_000,
    callbacks=EvalCallback("CartPole-v1", eval_freq=5_000, best_model_path="best_dqn.relepy"),
    log_interval=20,
)

best = DoubleDQN.load("best_dqn.relepy", "CartPole-v1")
print("best model:", best.evaluate(n_episodes=10))
