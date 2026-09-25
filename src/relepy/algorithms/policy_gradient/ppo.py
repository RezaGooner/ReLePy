"""Proximal Policy Optimization (Schulman et al., 2017), clipped-objective version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from relepy.algorithms.policy_gradient._base import OnPolicyAgent, OnPolicyConfig


@dataclass
class PPOConfig(OnPolicyConfig):
    """PPO hyperparameters.

    Attributes:
        batch_size: Mini-batch size (must be ``<= n_steps``).
        n_epochs: Optimization epochs per rollout.
        clip_range: Clipping parameter epsilon of the surrogate objective.
        target_kl: If set, stop the epochs early once the approximate KL exceeds
            ``1.5 * target_kl``.
    """

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    target_kl: Optional[float] = None

    def validate(self) -> None:
        super().validate()
        if self.batch_size < 1 or self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size must be in [1, n_steps={self.n_steps}], got {self.batch_size}"
            )
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.clip_range <= 0:
            raise ValueError("clip_range must be > 0")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be > 0 or None")


class PPO(OnPolicyAgent):
    """PPO with GAE. Supports ``Discrete`` and ``Box`` action spaces.

    Example:
        >>> agent = PPO("Pendulum-v1", n_steps=1024, seed=0)  # doctest: +SKIP
        >>> agent.fit(100_000)  # doctest: +SKIP
    """

    config_class = PPOConfig
    config: PPOConfig

    def _update(self) -> Dict[str, float]:
        cfg = self.config
        stats: Dict[str, list] = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": []}
        stop = False
        for _ in range(cfg.n_epochs):
            for batch in self.rollout_buffer.get(cfg.batch_size, self.device):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    batch.observations, batch.actions
                )
                adv = batch.advantages
                if cfg.normalize_advantage and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_ratio = log_prob - batch.old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = adv * ratio
                surr2 = adv * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch.returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
                self._optimize(loss)

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                    clip_frac = float(((ratio - 1.0).abs() > cfg.clip_range).float().mean())
                stats["pg"].append(policy_loss.item())
                stats["vf"].append(value_loss.item())
                stats["ent"].append(-entropy_loss.item())
                stats["kl"].append(approx_kl)
                stats["clip"].append(clip_frac)
                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {
            "train/policy_loss": float(np.mean(stats["pg"])),
            "train/value_loss": float(np.mean(stats["vf"])),
            "train/entropy": float(np.mean(stats["ent"])),
            "train/approx_kl": float(np.mean(stats["kl"])),
            "train/clip_fraction": float(np.mean(stats["clip"])),
        }
