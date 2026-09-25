"""The abstract base class every ReLePy agent derives from."""

from __future__ import annotations

import abc
import pickle
from collections import deque
from pathlib import Path
from typing import Any, ClassVar, Deque, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

import gymnasium as gym
import numpy as np

from relepy._version import __version__
from relepy.core.callbacks import Callback, CallbackList
from relepy.core.config import BaseConfig
from relepy.utils.evaluation import evaluate_policy
from relepy.utils.logger import Logger
from relepy.utils.seeding import set_seed
from relepy.utils.spaces import make_env

A = TypeVar("A", bound="BaseAgent")


class BaseAgent(abc.ABC):
    """Common interface: ``fit`` / ``predict`` / ``evaluate`` / ``save`` / ``load``.

    Args:
        env: A Gymnasium environment or its id (e.g. ``"CartPole-v1"``).
        config: A config object of type :attr:`config_class`. Defaults are used if omitted.
        seed: Seed for reproducibility (Python, NumPy, PyTorch, environment).
        verbose: ``0`` silent, ``1`` training logs.
        **config_overrides: Hyperparameters that override ``config`` (or the defaults), e.g.
            ``DQN("CartPole-v1", learning_rate=1e-3)``.

    Subclasses implement ``_setup``, ``_learn``, ``predict``, ``_get_state`` and ``_set_state``.
    """

    config_class: ClassVar[Type[BaseConfig]] = BaseConfig
    #: Hyperparameters forced by a subclass (e.g. ``DoubleDQN`` forces ``double_dqn=True``).
    _forced_config: ClassVar[Dict[str, Any]] = {}

    def __init__(
        self,
        env: Union[str, gym.Env],
        config: Optional[BaseConfig] = None,
        *,
        seed: Optional[int] = None,
        verbose: int = 1,
        **config_overrides: Any,
    ) -> None:
        self.config = self._resolve_config(config, config_overrides)
        self._env_id: Optional[str] = env if isinstance(env, str) else None
        self.env: gym.Env = make_env(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = seed
        self.rng: np.random.Generator = set_seed(seed)
        self.logger = Logger(verbose)

        self.num_timesteps = 0
        self.episode_rewards: Deque[float] = deque(maxlen=100)
        self.episode_lengths: Deque[int] = deque(maxlen=100)
        self._n_episodes = 0
        self._obs: Any = None
        self._ep_return = 0.0
        self._ep_len = 0
        self._needs_seed = seed is not None
        self._callbacks = CallbackList()

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

    @abc.abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """Return a picklable dict with everything needed to restore the learned policy."""

    @abc.abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Inverse of :meth:`_get_state`."""

    # ---------------------------------------------------------------- public API
    def fit(
        self,
        total_timesteps: int,
        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
        log_interval: int = 10,
    ) -> "BaseAgent":
        """Train for ``total_timesteps`` *additional* environment steps and return ``self``.

        ``log_interval`` counts episodes (tabular, DQN, REINFORCE: updates) or updates
        (A2C, PPO), see each algorithm's documentation.
        """
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
        self._callbacks = CallbackList(callbacks)
        if self._obs is None:
            self._reset_env()
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
        """Return ``(mean_return, std_return)`` over ``n_episodes`` on a separate environment."""
        if env is None:
            if self._env_id is None:
                raise ValueError(
                    "Pass an evaluation env: the agent was built from an env instance, so a "
                    "fresh copy cannot be created automatically."
                )
            env = self._env_id
        return evaluate_policy(self, make_env(env), n_episodes, deterministic, seed)

    def save(self, path: Union[str, Path]) -> None:
        """Save config, learned parameters and training progress (not the replay buffer)."""
        payload = {
            "relepy_version": __version__,
            "algorithm": type(self).__name__,
            "config": self.config.to_dict(),
            "num_timesteps": self.num_timesteps,
            "state": self._get_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(
        cls: Type[A],
        path: Union[str, Path],
        env: Union[str, gym.Env],
        *,
        seed: Optional[int] = None,
        verbose: int = 1,
        **config_overrides: Any,
    ) -> A:
        """Load an agent saved with :meth:`save`. Only load files you trust (pickle)."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["algorithm"] != cls.__name__:
            raise ValueError(
                f"File contains a {payload['algorithm']} agent, cannot load it as {cls.__name__}"
            )
        config = cls.config_class.from_dict(payload["config"])
        agent = cls(env, config, seed=seed, verbose=verbose, **config_overrides)
        agent._set_state(payload["state"])
        agent.num_timesteps = int(payload["num_timesteps"])
        return agent

    # ------------------------------------------------------- helpers for subclasses
    def _reset_env(self) -> Any:
        if self._needs_seed:
            obs, _ = self.env.reset(seed=self.seed)
            self.action_space.seed(self.seed)
            self._needs_seed = False
        else:
            obs, _ = self.env.reset()
        self._obs = obs
        self._ep_return = 0.0
        self._ep_len = 0
        return obs

    def _env_step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step the env, advance the timestep counter and track episode statistics.

        The caller is responsible for updating ``self._obs`` (or calling ``_reset_env``).
        """
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        self.num_timesteps += 1
        self._ep_return += reward
        self._ep_len += 1
        if terminated or truncated:
            self.episode_rewards.append(self._ep_return)
            self.episode_lengths.append(self._ep_len)
            self._n_episodes += 1
        return next_obs, reward, bool(terminated), bool(truncated), info

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(env={self.env}, timesteps={self.num_timesteps})"
