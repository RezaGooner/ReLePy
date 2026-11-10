# ReLePy

[![PyPI](https://img.shields.io/pypi/v/relepy.svg)](https://pypi.org/project/relepy/)
[![Python](https://img.shields.io/pypi/pyversions/relepy.svg)](https://pypi.org/project/relepy/)
[![CI](https://github.com/RezaGooner/ReLePy/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaGooner/ReLePy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**ReLePy** is a modular reinforcement learning library for Python with a small, consistent API:
build an agent, `fit` it, `predict` with it, `evaluate` it, `save` / `load` it. Every algorithm
has a typed, validated configuration object, so hyperparameters are explicit and experiments
are reproducible, from tabular Q-learning to SAC.

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

All algorithms support several parallel environments (`n_envs`).

## Parallel environments

```python
from relepy import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128, batch_size=256, seed=0)
agent.fit(100_000)
```

Pass an env id or a factory (`lambda: gym.make("CartPole-v1")`) when `n_envs > 1`. Timestep
budgets always count steps summed over all environments.

## Building agents by name

```python
import relepy

relepy.list_algorithms()                   # ['a2c', 'ddpg', 'double_dqn', 'dqn', ...]
agent = relepy.make("sac", "Pendulum-v1", seed=0)
agent = relepy.load("model.relepy", "Pendulum-v1")   # algorithm is read from the file
```

Your own agents can join the registry with `@relepy.register("my_algo")`; see
[`examples/custom_algorithm.py`](examples/custom_algorithm.py).

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

## Design

* **Components**: a `Collector` does all environment interaction; shared training-loop templates
  (`StepwiseAgent`, `OffPolicyAgent`, `OnPolicyAgent`) and loss functions
  (`relepy.core.losses`) keep each algorithm short. See [docs/index.md](docs/index.md).
* Works with any [Gymnasium](https://gymnasium.farama.org/) environment (id, factory or instance).
* `Discrete` observations are one-hot encoded for neural agents; `Box` observations are flattened.
* Time-limit truncation is handled correctly: value targets still bootstrap when an episode is
  truncated, and only true termination zeroes the bootstrap.
* **Safe model files**: `.relepy` files are ZIP archives of JSON and NumPy arrays; loading
  never executes code. Files from 0.1.x need `allow_legacy_pickle=True` once.

## Benchmarks

```bash
pip install -e ".[torch,benchmark]"
python benchmarks/run_benchmarks.py --suite quick --seeds 0 1 2
```

Results (CSV, learning curves, Markdown summary) are written to `benchmark_results/`.

## Development

```bash
pip install -e ".[torch,dev]"
ruff check src tests examples benchmarks
mypy src
pytest                    # fast tests
pytest -m slow            # learning tests (minutes)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding algorithms and making releases, and
[CHANGELOG.md](CHANGELOG.md) for what changed between versions.

## Roadmap

Prioritized replay and n-step returns, observation/reward normalization, CNN and recurrent
policies, more algorithms (classic DP/Monte-Carlo/bandits, Rainbow, TRPO, TQC, offline RL).

## License

MIT
