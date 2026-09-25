"""Shared code for off-policy continuous-control agents (DDPG, TD3, SAC)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from relepy.core.base import BaseAgent
from relepy.core.buffers import ReplayBuffer
from relepy.core.config import DeepConfig
from relepy.utils.spaces import make_obs_preprocessor
from relepy.utils.torch_utils import get_device


@dataclass
class OffPolicyConfig(DeepConfig):
    """Hyperparameters shared by DDPG, TD3 and SAC.

    Attributes:
        learning_rate: Critic (and, unless overridden, actor) learning rate.
        actor_learning_rate: Actor learning rate; ``None`` uses ``learning_rate``.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size per gradient step.
        learning_starts: Steps of uniform random actions before learning begins.
        train_freq: Do ``gradient_steps`` updates every ``train_freq`` environment steps.
        gradient_steps: Gradient steps per training phase.
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
        for name in ("buffer_size", "batch_size", "train_freq", "gradient_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.actor_learning_rate is not None and self.actor_learning_rate <= 0:
            raise ValueError("actor_learning_rate must be > 0 or None")


class OffPolicyAgent(BaseAgent):
    """Replay-buffer training loop for bounded ``Box`` action spaces.

    Networks always act in the normalized range ``[-1, 1]``; :meth:`_to_env_action` rescales
    to the environment's bounds. ``fit(..., log_interval=k)`` logs every ``k`` episodes.
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
        cfg = self.config
        self.device = get_device(cfg.device)
        self._obs_dim, self._preprocess = make_obs_preprocessor(self.observation_space)
        self.act_dim = int(np.prod(self.action_space.shape))
        self._low = self.action_space.low.reshape(-1).astype(np.float32)
        self._high = self.action_space.high.reshape(-1).astype(np.float32)
        self.buffer = ReplayBuffer(
            cfg.buffer_size, self._obs_dim, (self.act_dim,), np.float32, rng=self.rng
        )
        self._last_stats: Dict[str, float] = {}
        self._build()

    # ------------------------------------------------------------ subclass hooks
    @abc.abstractmethod
    def _build(self) -> None:
        """Create networks and optimizers."""

    @abc.abstractmethod
    def _policy_action(self, obs_vec: np.ndarray, deterministic: bool) -> np.ndarray:
        """Return a normalized action in ``[-1, 1]`` of shape ``(act_dim,)``."""

    @abc.abstractmethod
    def _train_step(self) -> Dict[str, float]:
        """Do one gradient step on a replay batch and return statistics to log."""

    # ------------------------------------------------------------------ helpers
    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        scaled = self._low + (action + 1.0) * 0.5 * (self._high - self._low)
        scaled = np.clip(scaled, self._low, self._high)
        return scaled.reshape(self.action_space.shape).astype(np.float32)

    def _obs_tensor(self, obs_vec: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _optimize(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> None:
        optimizer.zero_grad()
        loss.backward()
        if params is not None and self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
        optimizer.step()

    def predict(self, observation: Any, deterministic: bool = True) -> np.ndarray:
        action = self._policy_action(self._preprocess(observation), deterministic)
        return self._to_env_action(action)

    # ---------------------------------------------------------------- learning
    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        cfg = self.config
        end = self.num_timesteps + total_timesteps
        while self.num_timesteps < end:
            obs_vec = self._preprocess(self._obs)
            if self.num_timesteps < cfg.learning_starts:
                action = self.rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
            else:
                action = self._policy_action(obs_vec, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self._env_step(
                self._to_env_action(action)
            )
            self.buffer.add(obs_vec, action, reward, self._preprocess(next_obs), terminated)
            done = terminated or truncated
            if done:
                self._reset_env()
            else:
                self._obs = next_obs

            if (
                self.num_timesteps >= cfg.learning_starts
                and self.num_timesteps % cfg.train_freq == 0
                and len(self.buffer) >= cfg.batch_size
            ):
                for _ in range(cfg.gradient_steps):
                    self._last_stats = self._train_step()

            if not self._on_step():
                return
            if done and self._n_episodes % log_interval == 0:
                self._log_training(self._last_stats)
