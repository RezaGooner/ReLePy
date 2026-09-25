# Changelog

**Note on version numbers:** everything that was internally tracked as 0.1/0.2/0.3 during
development below is being released to PyPI as a single **0.1.0**. The "## 0.x" headings
below describe what changed at each development stage, kept for reference.

## 0.3.0
- **Vectorized training**: `PPO` and `A2C` accept `n_envs` in their config and step several
  environments in parallel (`relepy.envs.SyncVecEnv` / `make_vec_env`). Pass an env id or a
  factory callable (e.g. `lambda: gym.make("CartPole-v1")`) with `n_envs > 1`.
  Time-limit truncation is still handled correctly (bootstraps from `final_observation`).
- **Pickle-free save format**: `.relepy` files are now ZIP archives of JSON + `.npy` arrays
  (see `relepy.core.serialization`); loading never executes arbitrary code. Files saved by
  ReLePy 0.1/0.2 (pickle) are refused by default; pass `allow_legacy_pickle=True` to
  `Agent.load(...)` to opt in, then re-save with the new format.
- **Benchmark**: `examples/benchmark_vs_sb3.py` compares ReLePy's PPO against
  Stable-Baselines3 on the same task/budget (mean return and wall-clock time per run).
  Install with `pip install "relepy[benchmark]"`.
- `BaseAgent.evaluate()` now creates a fresh environment via the stored factory when the agent
  was built from an env id or factory (previously required calling with an explicit env in
  some cases); unchanged if you built the agent from an env instance.

## 0.2.0
- Added off-policy continuous-control algorithms: `DDPG`, `TD3`, `SAC` (with automatic entropy tuning).
- New networks in `relepy.core.policies`: `DeterministicActor`, `SquashedGaussianActor`, `ContinuousCritic`.
- Fixed a PyTorch `requires_grad` scalar-conversion warning in A2C, PPO and REINFORCE logging.
- `pytest` now skips slow learning tests by default; run them with `pytest -m slow`.

## 0.1.0
- Initial release: `QLearning`, `SARSA`, `ExpectedSARSA`, `DQN` (+ Double/Dueling), `REINFORCE`, `A2C`, `PPO`.
