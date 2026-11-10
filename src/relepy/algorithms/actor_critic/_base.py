"""Shared code for off-policy continuous-control agents (DDPG, TD3, SAC)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import numpy as np
from gymnasium import spaces

from relepy.core.off_policy import OffPolicyAgent, ReplayConfig


@dataclass
class OffPolicyConfig(ReplayConfig):
    """Hyperparameters shared by DDPG, TD3 and SAC.

    Attributes:
        learning_rate: Critic (and, unless overridden, actor) learning rate.
        actor_learning_rate: Actor learning rate; ``None`` uses ``learning_rate``.
        learning_starts: Steps of uniform random actions before learning begins.
        tau: Polyak coefficient for target networks (``1.0`` = hard copy).
    """

    learning_rate: float = 3e-4
    actor_learning_rate: Optional[float] = None
    hidden_sizes: Tuple[int, ...] = (256, 256)
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005

    def validate(self) -> None:
        super().validate()
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.actor_learning_rate is not None and self.actor_learning_rate <= 0:
            raise ValueError("actor_learning_rate must be > 0 or None")


class ContinuousOffPolicyAgent(OffPolicyAgent):
    """Replay-buffer agent for bounded ``Box`` action spaces.

    Networks always act in the normalized range ``[-1, 1]`` (that is also what is stored in the
    replay buffer); :meth:`_to_env_action` rescales to the environment's bounds.
    ``fit(..., log_interval=k)`` logs every ``k`` episodes.
    """

    config: OffPolicyConfig

    def _check_spaces(self) -> None:
        space = self.action_space
        if not isinstance(space, spaces.Box):
            raise TypeError(
                f"{type(self).__name__} requires a continuous (Box) action space, "
                f"got {type(space).__name__}. Use DQN/PPO for discrete actions."
            )
        if not (np.isfinite(space.low).all() and np.isfinite(space.high).all()):
            raise ValueError("Action bounds must be finite; wrap the environment to bound them.")

    def _setup(self) -> None:
        self._box = cast(spaces.Box, self.action_space)
        self.act_dim = int(np.prod(self._box.shape))
        self._low = self._box.low.reshape(-1).astype(np.float32)
        self._high = self._box.high.reshape(-1).astype(np.float32)
        super()._setup()

    def _buffer_action_spec(self) -> Tuple[Tuple[int, ...], type]:
        return (self.act_dim,), np.float32

    @property
    def actor_lr(self) -> float:
        return self.config.actor_learning_rate or self.config.learning_rate

    # ------------------------------------------------------------ subclass hook
    @abc.abstractmethod
    def _policy_actions(self, obs_vecs: np.ndarray, deterministic: bool) -> np.ndarray:
        """Return normalized actions in ``[-1, 1]`` of shape ``(batch, act_dim)``."""

    # ------------------------------------------------------------------ helpers
    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        """Normalized action(s) ``(..., act_dim)`` -> environment bounds and shape."""
        action = np.asarray(action, dtype=np.float32)
        scaled = self._low + (action + 1.0) * 0.5 * (self._high - self._low)
        scaled = np.clip(scaled, self._low, self._high)
        batch_shape = action.shape[:-1]
        return scaled.reshape(batch_shape + self._box.shape).astype(np.float32)

    def _select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < self.config.learning_starts:
            actions = self.rng.uniform(-1.0, 1.0, size=(self.n_envs, self.act_dim))
            actions = actions.astype(np.float32)
        else:
            actions = self._policy_actions(self._preprocess_batch(obs), deterministic=False)
        return self._to_env_action(actions), actions

    def predict(self, observation: Any, deterministic: bool = True) -> np.ndarray:
        action = self._policy_actions(self._preprocess(observation)[None], deterministic)[0]
        return self._to_env_action(action)
