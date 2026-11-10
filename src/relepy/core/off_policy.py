"""Template for replay-buffer (off-policy) deep RL agents: DQN, DDPG, TD3, SAC.

Requires PyTorch.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from relepy.core.buffers import ReplayBuffer
from relepy.core.collector import Transition
from relepy.core.config import DeepConfig
from relepy.core.loops import StepwiseAgent
from relepy.utils.spaces import make_batch_preprocessor, make_obs_preprocessor
from relepy.utils.torch_utils import get_device


@dataclass
class ReplayConfig(DeepConfig):
    """Settings shared by every replay-buffer algorithm.

    Attributes:
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size per gradient step.
        learning_starts: Environment steps collected before learning begins.
        train_freq: Run a training phase every ``train_freq`` environment steps (summed over
            all environments, so the ratio of gradient steps to samples does not depend on
            ``n_envs``).
        gradient_steps: Gradient steps per training phase.
    """

    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 4
    gradient_steps: int = 1

    def validate(self) -> None:
        super().validate()
        for name in ("buffer_size", "batch_size", "train_freq", "gradient_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")


class OffPolicyAgent(StepwiseAgent):
    """Stores every transition in a :class:`ReplayBuffer` and trains on schedule.

    Subclasses implement :meth:`_build` (networks, optimizers), :meth:`_buffer_action_spec`,
    :meth:`_select_actions` and :meth:`_train_step` (one gradient step, returns stats).
    The replay buffer is not saved by :meth:`save`.
    """

    config: ReplayConfig

    def _setup(self) -> None:
        cfg = self.config
        self.device = get_device(cfg.device)
        self._obs_dim, self._preprocess = make_obs_preprocessor(self.observation_space)
        _, self._preprocess_batch = make_batch_preprocessor(self.observation_space)
        action_shape, action_dtype = self._buffer_action_spec()
        self.buffer = ReplayBuffer(
            cfg.buffer_size, self._obs_dim, action_shape, action_dtype, rng=self.rng
        )
        self._last_stats: Dict[str, float] = {}
        self._steps_since_train = 0
        self._build()

    # ------------------------------------------------------------ subclass hooks
    @abc.abstractmethod
    def _build(self) -> None:
        """Create networks and optimizers."""

    @abc.abstractmethod
    def _buffer_action_spec(self) -> Tuple[Tuple[int, ...], type]:
        """``(shape, dtype)`` of one action as stored in the replay buffer."""

    @abc.abstractmethod
    def _train_step(self) -> Dict[str, float]:
        """One gradient step on a replay batch; returns statistics to log."""

    # ---------------------------------------------------------------- template
    def _observe(self, transition: Transition) -> None:
        cfg = self.config
        self.buffer.extend(
            self._preprocess_batch(transition.obs),
            transition.actions,
            transition.rewards,
            self._preprocess_batch(transition.next_obs),
            transition.terminated,  # truncation still bootstraps
        )
        self._steps_since_train += self.n_envs
        while self._steps_since_train >= cfg.train_freq:
            self._steps_since_train -= cfg.train_freq
            if self.num_timesteps >= cfg.learning_starts and len(self.buffer) >= cfg.batch_size:
                for _ in range(cfg.gradient_steps):
                    self._last_stats = self._train_step()

    def _train_stats(self) -> Dict[str, Any]:
        return dict(self._last_stats)

    # ------------------------------------------------------------------ helpers
    def _obs_tensor(self, obs_vecs: np.ndarray) -> torch.Tensor:
        """Batch of preprocessed observations -> float tensor on the agent's device."""
        return torch.as_tensor(obs_vecs, dtype=torch.float32, device=self.device)

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
