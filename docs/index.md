# ReLePy documentation

## Concepts

* **Agent**: every algorithm derives from `relepy.BaseAgent` and exposes
  `fit`, `predict`, `evaluate`, `save`, `load`.
* **Config**: every agent has a `*Config` dataclass. Pass a config object and/or keyword
  overrides: `DQN(env, DQNConfig(...), seed=0, batch_size=128)`. Configs are validated when
  created and can be saved to and loaded from JSON.
* **Environments**: an env id, a factory callable, a `gymnasium.Env` instance (single env) or
  a `relepy.SyncVecEnv`. `n_envs` (in every config) runs several environments in parallel.
* **Callback**: hooks into `fit` (`on_training_start`, `on_step`, `on_training_end`).
* **Logger**: `agent.logger.history` holds one dict per logged row.
* **Registry**: `relepy.make(name, env, ...)`, `relepy.list_algorithms()`,
  `relepy.load(path, env)`, `@relepy.register(name)`.

## Architecture

```
               fit / predict / evaluate / save / load        (BaseAgent)
                                  │
   ┌──────────────────────────────┼─────────────────────────────────┐
   │ Training-loop templates      │                                 │
   │   StepwiseAgent  ── tabular TD (QLearning, SARSA, ...)         │
   │     └ OffPolicyAgent ── DQN family, DDPG, TD3, SAC             │
   │   OnPolicyAgent  ── A2C, PPO       ActorCriticAgent ── REINFORCE
   └──────────────────────────────┬─────────────────────────────────┘
                                  │ uses
   Collector ── SyncVecEnv     Buffers (Replay, Rollout+GAE)     Losses (plain functions)
   Policies (MLP Q-net, actor-critic, deterministic / squashed-Gaussian actors, critics)
   Serialization (ZIP of JSON + .npy)    Registry    Callbacks    Logger
```

| Module | Needs PyTorch | Role |
|---|---|---|
| `relepy.core.base` | no | `BaseAgent`: config resolution, env + collector, fit/evaluate/save/load |
| `relepy.core.collector` | no | steps the vector env; counts timesteps and episodes; `Transition` |
| `relepy.core.loops` | no | `StepwiseAgent` template |
| `relepy.core.registry` | no | names -> agent classes, `make`, `load`, `register` |
| `relepy.core.serialization` | no | pickle-free `.relepy` archives |
| `relepy.envs` | no | `SyncVecEnv`, `make_vec_env`, `GridWorldEnv` |
| `relepy.core.off_policy` | yes | `ReplayConfig`, `OffPolicyAgent` template |
| `relepy.core.on_policy` | yes | `ActorCriticAgent`, `OnPolicyAgent` templates |
| `relepy.core.buffers` | yes | `ReplayBuffer`, `RolloutBuffer` (GAE) |
| `relepy.core.losses` | yes | DQN, REINFORCE, A2C, PPO, DDPG/TD3, SAC objectives |
| `relepy.core.policies` | yes | neural network modules |

### Vectorized environments

`SyncVecEnv` resets finished environments in the same step and returns the true last
observation of the finished episode separately, so the semantics do not depend on the
Gymnasium version. Each `Transition` therefore holds `obs`, `actions`, `rewards`, `next_obs`
(true next observation), `terminated` and `truncated`, all with a leading `n_envs` axis.

* Timestep budgets (`fit(total_timesteps)`, `learning_starts`, `train_freq`,
  `target_update_interval`, epsilon schedules) count steps **summed over all environments**.
* On-policy agents collect `n_steps` per environment, i.e. `n_steps * n_envs` samples per update.
* Time-limit truncation bootstraps: off-policy agents store only `terminated` as the done flag;
  on-policy agents add `gamma * V(final_obs)` to the reward of truncated steps.

### Saved files

A `.relepy` file is a ZIP archive with `metadata.json` (algorithm, config, versions, progress),
`state.json` (the structure of the learned state) and `arrays/*.npy`. Arrays are read with
`allow_pickle=False`, so loading never executes code from the file. Files from 0.1.x (pickle)
need `allow_legacy_pickle=True`; re-save them to convert.

## Adding a new algorithm

Pick the template that matches the algorithm:

* **Learns every step** (tabular TD, ...): subclass `StepwiseAgent`, implement `_setup`,
  `predict`, `_select_actions(obs) -> (env_actions, recorded_actions)` and
  `_observe(transition)`.
* **Replay buffer** (DQN-like, actor-critic): subclass `OffPolicyAgent`, implement `_build`,
  `_buffer_action_spec`, `_select_actions`, `_train_step` and `predict`.
* **Rollouts + GAE**: subclass `OnPolicyAgent` and implement `_update()` using
  `self.rollout_buffer.get(batch_size, device)`.
* **Anything else**: subclass `BaseAgent` and write `_learn` yourself, stepping environments
  with `self.collector.step(actions)` and calling `self._on_step()` after each step (it returns
  `False` when a callback asks to stop).

Then list learned attributes in `_state_attrs`, put the objective in `relepy.core.losses`, and
register the class. A complete example (Double Q-learning in ~40 lines) is in
[`examples/custom_algorithm.py`](../examples/custom_algorithm.py); the checklist for a pull
request is in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Benchmarks

```bash
pip install -e ".[torch,benchmark]"
python benchmarks/run_benchmarks.py --suite quick --seeds 0 1 2
```

Writes `results.csv`, `curves.csv`, `summary.md` and `curves.png` to `benchmark_results/`.

## Releasing

See the *Releasing* section of [CONTRIBUTING.md](../CONTRIBUTING.md). In short: bump
`src/relepy/_version.py`, update the changelog, push a `vX.Y.Z` tag.
