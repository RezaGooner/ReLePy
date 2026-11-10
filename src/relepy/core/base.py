"""The abstract base class every ReLePy agent derives from."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Callable, ClassVar, Deque, Dict, Optional, Sequence, Tuple, Type, TypeVar
from typing import Union

import gymnasium as gym
import numpy as np

from relepy._version import __version__
from relepy.core.callbacks import Callback, CallbackList
from relepy.core.collector import Collector
from relepy.core.config import BaseConfig
from relepy.core.serialization import load_archive, save_archive
from relepy.envs.vec_env import EnvLike, SyncVecEnv, env_factory, make_vec_env
from relepy.utils.evaluation import evaluate_policy
from relepy.utils.logger import Logger
from relepy.utils.seeding import set_seed
from relepy.utils.spaces import make_env

A = TypeVar("A", bound="BaseAgent")


def _is_tensor(obj: Any) -> bool:
    return type(obj).__module__.startswith("torch") and hasattr(obj, "copy_")


class BaseAgent(abc.ABC):
    """Common interface: ``fit`` / ``predict`` / ``evaluate`` / ``save`` / ``load``.

    Args:
        env: A Gymnasium id (e.g. ``"CartPole-v1"``), a factory callable returning a new env,
            a ``gymnasium.Env`` instance (only with ``n_envs=1``) or a :class:`SyncVecEnv`.
        config: A config object of type :attr:`config_class`. Defaults are used if omitted.
        seed: Seed for reproducibility (Python, NumPy, PyTorch, environments).
        verbose: ``0`` silent, ``1`` training logs.
        **config_overrides: Hyperparameters that override ``config`` (or the defaults), e.g.
            ``DQN("CartPole-v1", learning_rate=1e-3, n_envs=4)``.

    Architecture: the agent owns a :class:`~relepy.core.collector.Collector` (all environment
    interaction), a :class:`~relepy.utils.logger.Logger` and callbacks. Subclasses implement
    ``_setup`` (build networks/tables/buffers), ``_learn`` (the training loop, usually inherited
    from a template such as :class:`~relepy.core.loops.StepwiseAgent`) and ``predict``. Listing
    attribute names in :attr:`_state_attrs` is enough to make them saved and loaded.
    """

    config_class: ClassVar[Type[BaseConfig]] = BaseConfig
    #: Hyperparameters forced by a subclass (e.g. ``DoubleDQN`` forces ``double_dqn=True``).
    _forced_config: ClassVar[Dict[str, Any]] = {}
    #: Attributes persisted by :meth:`save`: modules/optimizers (``state_dict``), tensors,
    #: NumPy arrays or plain values.
    _state_attrs: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        env: Union[EnvLike, SyncVecEnv],
        config: Optional[BaseConfig] = None,
        *,
        seed: Optional[int] = None,
        verbose: int = 1,
        **config_overrides: Any,
    ) -> None:
        self.config = self._resolve_config(config, config_overrides)
        self._env_factory: Optional[Callable[[], gym.Env]] = (
            None if isinstance(env, SyncVecEnv) else env_factory(env)
        )
        self.vec_env = make_vec_env(env, self.config.n_envs)
        self.n_envs = self.vec_env.n_envs
        self.env: gym.Env = self.vec_env.envs[0]
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space
        self.seed = seed
        self.rng: np.random.Generator = set_seed(seed)
        self.logger = Logger(verbose)
        self.collector = Collector(self.vec_env, seed=seed)
        self._callbacks = CallbackList()
        self._last_log_bucket = 0

        self._check_spaces()
        self._setup()

    # ------------------------------------------------------------ configuration
    @classmethod
    def _resolve_config(cls, config: Optional[BaseConfig], overrides: Dict[str, Any]) -> Any:
        if config is None:
            config = cls.config_class()
        elif not isinstance(config, cls.config_class):
            raise TypeError(
                f"{cls.__name__} expects a {cls.config_class.__name__}, "
                f"got {type(config).__name__}"
            )
        merged = {**overrides, **cls._forced_config}
        return config.update(**merged) if merged else config

    # ------------------------------------------------------------ progress (collector)
    @property
    def num_timesteps(self) -> int:
        """Environment steps taken so far, summed over all environments."""
        return self.collector.num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value: int) -> None:
        self.collector.num_timesteps = int(value)

    @property
    def episode_rewards(self) -> Deque[float]:
        """Returns of the most recent (up to 100) finished training episodes."""
        return self.collector.episode_rewards

    @property
    def episode_lengths(self) -> Deque[int]:
        return self.collector.episode_lengths

    @property
    def _n_episodes(self) -> int:
        return self.collector.n_episodes

    # ------------------------------------------------------------ subclass hooks
    def _check_spaces(self) -> None:
        """Validate observation/action spaces. Raise ``TypeError`` if unsupported."""

    @abc.abstractmethod
    def _setup(self) -> None:
        """Create networks, tables, buffers, optimizers."""

    @abc.abstractmethod
    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        """Run the training loop for ``total_timesteps`` additional steps."""

    @abc.abstractmethod
    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """Return an action for a single (unbatched) observation."""

    def _get_state(self) -> Dict[str, Any]:
        """Learned state to save. Default: every attribute named in :attr:`_state_attrs`."""
        state: Dict[str, Any] = {}
        for name in self._state_attrs:
            obj = getattr(self, name)
            state[name] = obj.state_dict() if hasattr(obj, "state_dict") else obj
        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Inverse of :meth:`_get_state`."""
        for name in self._state_attrs:
            if name not in state:
                raise ValueError(f"Saved state has no entry {name!r}")
            value, obj = state[name], getattr(self, name)
            if hasattr(obj, "load_state_dict"):
                obj.load_state_dict(value)
            elif _is_tensor(obj):
                import torch

                with torch.no_grad():
                    obj.copy_(torch.as_tensor(value).to(obj.device))
            elif isinstance(obj, np.ndarray):
                arr = np.asarray(value, dtype=obj.dtype)
                if arr.shape != obj.shape:
                    raise ValueError(
                        f"Saved {name!r} has shape {arr.shape}, this agent needs {obj.shape}"
                    )
                setattr(self, name, arr.copy())
            else:
                setattr(self, name, value)

    # ---------------------------------------------------------------- public API
    def fit(
        self,
        total_timesteps: int,
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
        log_interval: int = 10,
    ) -> "BaseAgent":
        """Train for ``total_timesteps`` *additional* environment steps and return ``self``.

        ``log_interval`` counts finished episodes for step-wise agents (tabular, DQN, DDPG,
        TD3, SAC) and updates for A2C, PPO and REINFORCE. Callbacks are called after every
        vector step (i.e. every ``n_envs`` timesteps).
        """
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
        self._callbacks = CallbackList(callbacks)
        self.collector.ensure_started()
        self._callbacks.on_training_start(self)
        try:
            self._learn(int(total_timesteps), max(1, int(log_interval)))
        finally:
            self._callbacks.on_training_end(self)
        return self

    def evaluate(
        self,
        env: Optional[Union[str, gym.Env]] = None,
        n_episodes: int = 10,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Return ``(mean_return, std_return)`` over ``n_episodes`` on a separate environment.

        Without ``env``, a fresh copy is created when the agent was built from an id or factory.
        """
        if env is None:
            if self._env_factory is None:
                raise ValueError(
                    "Pass an evaluation env: the agent was built from an env instance, so a "
                    "fresh copy cannot be created automatically."
                )
            eval_env = self._env_factory()
        else:
            eval_env = make_env(env)
        return evaluate_policy(self, eval_env, n_episodes, deterministic, seed)

    def save(self, path: Union[str, Path]) -> None:
        """Save config, learned parameters and progress as a pickle-free ``.relepy`` archive.

        Replay buffers are not saved.
        """
        metadata = {
            "relepy_version": __version__,
            "algorithm": type(self).__name__,
            "config": self.config.to_dict(),
            "num_timesteps": self.num_timesteps,
            "n_episodes": self._n_episodes,
        }
        save_archive(path, metadata, self._get_state())

    @classmethod
    def load(
        cls: Type[A],
        path: Union[str, Path],
        env: Union[EnvLike, SyncVecEnv],
        *,
        seed: Optional[int] = None,
        verbose: int = 1,
        allow_legacy_pickle: bool = False,
        **config_overrides: Any,
    ) -> A:
        """Load an agent saved with :meth:`save`.

        ``allow_legacy_pickle=True`` also accepts files written by ReLePy 0.1 (pickle): only
        use it for files you trust, then re-save them in the new format.
        """
        metadata, state = load_archive(path, allow_legacy_pickle=allow_legacy_pickle)
        if metadata["algorithm"] != cls.__name__:
            raise ValueError(
                f"File contains a {metadata['algorithm']} agent, cannot load it as {cls.__name__}"
            )
        config = cls.config_class.from_dict(metadata["config"])
        if (
            config.n_envs > 1
            and "n_envs" not in config_overrides
            and isinstance(env, gym.Env)
        ):
            # An env instance cannot be copied: fall back to a single env (e.g. for inference).
            try:
                config = config.update(n_envs=1)
            except ValueError as exc:
                raise ValueError(
                    f"This agent was trained with n_envs={config.n_envs} and its settings are "
                    f"not valid with a single env ({exc}). Pass an env id or factory instead "
                    "of an env instance, or override the conflicting hyperparameters."
                ) from exc
        agent = cls(env, config, seed=seed, verbose=verbose, **config_overrides)
        agent._set_state(state)
        agent.num_timesteps = int(metadata["num_timesteps"])
        agent.collector.n_episodes = int(metadata.get("n_episodes", 0))
        return agent

    # ------------------------------------------------------- helpers for subclasses
    def _on_step(self) -> bool:
        return self._callbacks.on_step(self)

    def _log_training(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if self.episode_rewards:
            self.logger.record("rollout/ep_rew_mean", float(np.mean(self.episode_rewards)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(self.episode_lengths)))
        self.logger.record("time/episodes", self._n_episodes)
        for key, value in (extra or {}).items():
            self.logger.record(key, value)
        self.logger.dump(self.num_timesteps)

    def _log_every_episodes(
        self, log_interval: int, extra: Callable[[], Optional[Dict[str, Any]]]
    ) -> None:
        """Log once each time another ``log_interval`` episodes have finished."""
        bucket = self._n_episodes // log_interval
        if bucket > self._last_log_bucket:
            self._last_log_bucket = bucket
            self._log_training(extra())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(env={self.env}, n_envs={self.n_envs}, "
            f"timesteps={self.num_timesteps})"
        )
