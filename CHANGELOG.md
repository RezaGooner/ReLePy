# Changelog

All notable changes to ReLePy are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/) (before 1.0, minor versions may
contain breaking changes, which are listed explicitly).

## [Unreleased]

## [0.2.0] - 2026-11-10

A redesign of the internals around reusable components. The public API
(`fit` / `predict` / `evaluate` / `save` / `load`, configs, callbacks) is unchanged.

### Added
- **Component architecture**: a `Collector` (`relepy.core.collector`) is now the only component
  that steps environments; training-loop templates `StepwiseAgent` (`relepy.core.loops`),
  `OffPolicyAgent` (`relepy.core.off_policy`) and `OnPolicyAgent` (`relepy.core.on_policy`)
  are shared by all algorithms, and every loss lives as a plain function in
  `relepy.core.losses`. New algorithms need far less code (see `examples/custom_algorithm.py`).
- **Vectorized environments for every algorithm**: `n_envs` in any config steps several
  environments in parallel (`relepy.SyncVecEnv`, `relepy.make_vec_env`). Pass an env id or a
  factory such as `lambda: gym.make("CartPole-v1")`. Auto-reset semantics are the same on every
  Gymnasium version and time-limit truncation still bootstraps from the true final observation.
- **Algorithm registry**: `relepy.make("ppo", env, ...)`, `relepy.list_algorithms()`,
  `relepy.get_algorithm(name)`, the `@relepy.register(name)` decorator for custom agents and
  `relepy.load(path, env)`, which reads the algorithm from the file.
- **Pickle-free model files**: `.relepy` files are ZIP archives of JSON and `.npy` arrays
  (`relepy.core.serialization`), so loading never executes code from the file.
- `PPOConfig.clip_range_vf` (value-function clipping) and `SACConfig.alpha_learning_rate`.
- Automatic persistence: listing attribute names in `_state_attrs` is enough for `save`/`load`.
- `benchmarks/run_benchmarks.py`: multi-seed benchmark writing CSV results, learning curves and
  a Markdown summary; a manually triggered GitHub Actions workflow runs it.
- CI: lint (ruff, mypy), tests on Python 3.10 - 3.13, a job without PyTorch, and a package
  build check. Releases are published to PyPI from version tags with Trusted Publishing.
- Python 3.13 support.

### Changed
- **Breaking:** Python 3.9 is no longer supported (end of life since October 2025); ReLePy
  now requires Python 3.10 or newer.
- **Breaking:** files saved by 0.1.x (pickle) are refused by default. Load them once with
  `Agent.load(path, env, allow_legacy_pickle=True)` (only for files you trust) and save again.
- **Breaking:** `RolloutBuffer` arrays have shape `(n_steps, n_envs, ...)`.
- `PPOConfig.batch_size` is validated against `n_steps * n_envs`.
- PPO with `target_kl` now stops *before* the gradient step that would exceed the limit
  (previously after it); the logged `train/stopped_early` shows when this happens.
- `DQN` and the off-policy agents count `train_freq` and `target_update_interval` in
  environment steps summed over all environments.
- `Config.from_json` treats `Path` objects always as files.
- Callbacks are called after every vector step (every `n_envs` timesteps).
- Loading an agent trained with `n_envs > 1` together with a single env instance falls back to
  one environment (useful for inference).

### Removed
- The internal helpers `BaseAgent._env_step` and `BaseAgent._reset_env` (use the collector).

## [0.1.0]

### Added
- First release: `QLearning`, `SARSA`, `ExpectedSARSA`, `DQN` (+ `DoubleDQN`, `DuelingDQN`),
  `REINFORCE`, `A2C`, `PPO`, `DDPG`, `TD3`, `SAC`.
- Typed, validated, JSON-serializable configuration objects for every algorithm.
- Callbacks (`EvalCallback`, `CheckpointCallback`, `StopOnRewardThreshold`) and an in-memory
  logger.

[Unreleased]: https://github.com/RezaGooner/ReLePy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RezaGooner/ReLePy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/RezaGooner/ReLePy/releases/tag/v0.1.0
