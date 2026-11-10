"""Proximal Policy Optimization (Schulman et al., 2017), clipped-objective version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from relepy.core.losses import ppo_loss
from relepy.core.on_policy import OnPolicyAgent, OnPolicyConfig


@dataclass
class PPOConfig(OnPolicyConfig):
    """PPO hyperparameters.

    Attributes:
        batch_size: Mini-batch size (must be ``<= n_steps * n_envs``).
        n_epochs: Optimization epochs per rollout.
        clip_range: Clipping parameter epsilon of the surrogate objective.
        clip_range_vf: If set, also clip the value-function update (relative to the old
            values) to this range.
        target_kl: If set, stop the update early, *before* the gradient step, once the
            approximate KL divergence exceeds ``1.5 * target_kl``.
    """

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    target_kl: Optional[float] = None

    def validate(self) -> None:
        super().validate()
        n_samples = self.n_steps * self.n_envs
        if self.batch_size < 1 or self.batch_size > n_samples:
            raise ValueError(
                f"batch_size must be in [1, n_steps * n_envs = {n_samples}], "
                f"got {self.batch_size}"
            )
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.clip_range <= 0:
            raise ValueError("clip_range must be > 0")
        if self.clip_range_vf is not None and self.clip_range_vf <= 0:
            raise ValueError("clip_range_vf must be > 0 or None")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 or None")


class PPO(OnPolicyAgent):
    """PPO with GAE. Supports ``Discrete`` and ``Box`` action spaces and ``n_envs > 1``.

    Example:
        >>> agent = PPO("CartPole-v1", n_envs=8, n_steps=128, batch_size=256)  # doctest: +SKIP
        >>> agent.fit(100_000)  # doctest: +SKIP
    """

    config_class = PPOConfig
    config: PPOConfig

    def _update(self) -> Dict[str, float]:
        cfg = self.config
        history: Dict[str, List[float]] = {}
        for _ in range(cfg.n_epochs):
            for batch in self.rollout_buffer.get(cfg.batch_size, self.device):
                loss, stats = ppo_loss(
                    self.policy, batch, cfg.clip_range, cfg.value_coef, cfg.entropy_coef,
                    cfg.normalize_advantage, cfg.clip_range_vf,
                )
                for key, value in stats.items():
                    history.setdefault(key, []).append(value)
                if cfg.target_kl is not None and stats["train/approx_kl"] > 1.5 * cfg.target_kl:
                    return self._summarize(history, stopped_early=True)
                self._optimize(loss)
        return self._summarize(history, stopped_early=False)

    @staticmethod
    def _summarize(history: Dict[str, List[float]], stopped_early: bool) -> Dict[str, float]:
        out = {key: float(np.mean(values)) for key, values in history.items()}
        out["train/stopped_early"] = float(stopped_early)
        return out
