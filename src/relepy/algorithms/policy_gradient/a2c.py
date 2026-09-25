"""Advantage Actor-Critic (Mnih et al., 2016), synchronous single-environment version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch.nn.functional as F

from relepy.algorithms.policy_gradient._base import OnPolicyAgent, OnPolicyConfig


@dataclass
class A2CConfig(OnPolicyConfig):
    """A2C hyperparameters. See :class:`OnPolicyConfig` for the shared ones."""

    learning_rate: float = 7e-4
    n_steps: int = 16
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01


class A2C(OnPolicyAgent):
    """A2C: one gradient step per rollout of ``n_steps`` steps, with GAE advantages.

    Supports ``Discrete`` and ``Box`` action spaces.
    """

    config_class = A2CConfig
    config: A2CConfig

    def _update(self) -> Dict[str, float]:
        cfg = self.config
        batch = next(self.rollout_buffer.get(None, self.device))
        values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, batch.actions)
        adv = batch.advantages
        if cfg.normalize_advantage and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(adv * log_prob).mean()
        value_loss = F.mse_loss(values, batch.returns)
        entropy_loss = -entropy.mean()
        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
        self._optimize(loss)
        return {
            "train/policy_loss": policy_loss.item(),
            "train/value_loss": value_loss.item(),
            "train/entropy": -entropy_loss.item(),
        }
