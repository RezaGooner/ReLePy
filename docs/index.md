# ReLePy documentation

## Concepts

* **Agent** – every algorithm derives from `relepy.BaseAgent` and exposes
  `fit`, `predict`, `evaluate`, `save`, `load`.
* **Config** – every agent has a `*Config` dataclass. Pass a config object and/or keyword
  overrides: `DQN(env, DQNConfig(...), seed=0, batch_size=128)`.
* **Callback** – hooks into `fit` (`on_training_start`, `on_step`, `on_training_end`).
* **Logger** – `agent.logger.history` holds one dict per logged row.

## Adding a new algorithm

1. Create a `MyConfig(DeepConfig)` dataclass (or `BaseConfig`) and override `validate`.
2. Subclass `BaseAgent`, set `config_class = MyConfig` and implement
   `_setup`, `_learn`, `predict`, `_get_state`, `_set_state`.
3. Inside `_learn` use `self._env_step(action)`, `self._reset_env()`, `self._on_step()`
   (returns `False` when a callback asks to stop) and `self._log_training({...})`.
4. Register the class in `relepy/__init__.py` (`_LAZY_ATTRS`) and add a test in `tests/`.

## Releasing

```bash
python -m build
twine check dist/*
twine upload --repository testpypi dist/*   # try TestPyPI first
twine upload dist/*
```

Bump the version in `src/relepy/_version.py` and tag the commit (`git tag v0.1.0`).
