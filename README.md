# ReLePy

**ReLePy** is a modular reinforcement learning library for Python with a small, consistent API:
build an agent, `fit` it, `predict` with it, `evaluate` it, `save` / `load` it. Every algorithm
has a typed, validated configuration object, so hyperparameters are explicit and experiments
are reproducible.

```python
from relepy import DQN

agent = DQN("CartPole-v1", learning_rate=1e-3, double_dqn=True, seed=42)
agent.fit(total_timesteps=30_000)

mean_reward, std_reward = agent.evaluate(n_episodes=10)
action = agent.predict(observation)          # single observation in, action out
agent.save("cartpole.relepy")
```

## Installation

```bash
pip install relepy            # tabular methods (NumPy + Gymnasium)
pip install "relepy[torch]"   # + deep RL algorithms (PyTorch)
```

From source: `pip install -e ".[torch,dev]"`.

## Algorithms

| Family | Algorithms | Observations | Actions |
|---|---|---|---|
| Tabular | `QLearning`, `SARSA`, `ExpectedSARSA` | Discrete | Discrete |
| Value-based | `DQN`, `DoubleDQN`, `DuelingDQN` | Box / Discrete | Discrete |
| Policy gradient | `REINFORCE`, `A2C`, `PPO` | Box / Discrete | Discrete / Box |
| Off-policy actor-critic | `DDPG`, `TD3`, `SAC` | Box / Discrete | Box (bounded) |

Planned: vectorized environments, prioritized replay, n-step returns, a safer model format.

## Hyperparameters

Pass them as keyword arguments or as a config object (validated on creation, serializable to JSON):

```python
from relepy import PPO, PPOConfig

config = PPOConfig(n_steps=1024, batch_size=64, clip_range=0.2, hidden_sizes=(128, 128))
config.to_json("ppo.json")                       # reproducible experiments
agent = PPO("Pendulum-v1", config, seed=0)
agent = PPO("Pendulum-v1", config, seed=0, n_epochs=5)  # keyword arguments override the config
```

## Callbacks

```python
from relepy import DQN, CheckpointCallback, EvalCallback

agent.fit(
    50_000,
    callbacks=[
        EvalCallback("CartPole-v1", eval_freq=5_000, best_model_path="best.relepy",
                     reward_threshold=475),
        CheckpointCallback(save_freq=10_000, save_dir="checkpoints"),
    ],
)
```

Write your own by subclassing `relepy.Callback`; return `False` from `on_step` to stop training.
Training statistics are kept in `agent.logger.history` (a list of dicts).

## Design notes

* Works with any [Gymnasium](https://gymnasium.farama.org/) environment (pass an id or an instance).
* `Discrete` observations are one-hot encoded for neural agents; `Box` observations are flattened.
* Time-limit truncation is handled correctly: value targets still bootstrap when an episode is
  truncated, and only true termination zeroes the bootstrap.
* Saved files (`.relepy`) use `pickle`: only load files you trust.
* Currently single-environment training (no vectorized environments yet).

## Development

```bash
pip install -e ".[torch,dev]"
ruff check src tests
pytest                    # fast tests
pytest -m slow            # learning tests (minutes)
```

## Citation

A paper is in preparation. Until then, please cite the repository.

## License

MIT
