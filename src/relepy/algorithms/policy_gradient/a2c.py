"""Advantage Actor-Critic (Mnih et al., 2016), synchronous version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from relepy.core.losses import a2c_loss
from relepy.core.on_policy import OnPolicyAgent, OnPolicyConfig


@dataclass
class A2CConfig(OnPolicyConfig):
    """A2C hyperparameters. See :class:`~relepy.core.on_policy.OnPolicyConfig`."""

    learning_rate: float = 7e-4
    n_steps: int = 16
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01


class A2C(OnPolicyAgent):
    """A2C: one gradient step per rollout of ``n_steps * n_envs`` samples, GAE advantages.

    Supports ``Discrete`` and ``Box`` action spaces and ``n_envs > 1``.
    """

    config_class = A2CConfig
    config: A2CConfig

    def _update(self) -> Dict[str, float]:
        cfg = self.config
        batch = next(self.rollout_buffer.get(None, self.device))
        loss, stats = a2c_loss(
            self.policy, batch, cfg.value_coef, cfg.entropy_coef, cfg.normalize_advantage
        )
        self._optimize(loss)
        return stats
