"""Soft Actor-Critic (Haarnoja et al., 2018) with automatic entropy tuning."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from relepy.algorithms.actor_critic._base import ContinuousOffPolicyAgent, OffPolicyConfig
from relepy.core.losses import sac_actor_loss, sac_critic_loss, sac_temperature_loss
from relepy.core.policies import ContinuousCritic, SquashedGaussianActor
from relepy.utils.torch_utils import polyak_update


@dataclass
class SACConfig(OffPolicyConfig):
    """SAC hyperparameters.

    Attributes:
        auto_entropy: Learn the entropy temperature alpha to match ``target_entropy``.
        initial_alpha: Initial (or, if ``auto_entropy=False``, fixed) temperature.
        target_entropy: Desired policy entropy; ``None`` uses ``-action_dim``.
        alpha_learning_rate: Learning rate of the temperature; ``None`` uses ``learning_rate``.
    """

    auto_entropy: bool = True
    initial_alpha: float = 1.0
    target_entropy: Optional[float] = None
    alpha_learning_rate: Optional[float] = None

    def validate(self) -> None:
        super().validate()
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha must be > 0")
        if self.alpha_learning_rate is not None and self.alpha_learning_rate <= 0:
            raise ValueError("alpha_learning_rate must be > 0 or None")


class SAC(ContinuousOffPolicyAgent):
    """SAC for bounded continuous action spaces.

    Example:
        >>> agent = SAC("Pendulum-v1", seed=0)  # doctest: +SKIP
        >>> agent.fit(15_000)  # doctest: +SKIP
    """

    config_class = SACConfig
    config: SACConfig
    _state_attrs = (
        "actor", "critic", "critic_target",
        "actor_optimizer", "critic_optimizer", "alpha_optimizer", "log_alpha",
    )

    def _build(self) -> None:
        cfg = self.config
        self.actor = SquashedGaussianActor(
            self._obs_dim, self.act_dim, cfg.hidden_sizes, cfg.activation
        ).to(self.device)
        self.critic = ContinuousCritic(
            self._obs_dim, self.act_dim, cfg.hidden_sizes, cfg.activation, n_critics=2
        ).to(self.device)
        self.critic_target = deepcopy(self.critic).eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            math.log(cfg.initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self._target_entropy = (
            cfg.target_entropy if cfg.target_entropy is not None else -float(self.act_dim)
        )
        alpha_lr = cfg.alpha_learning_rate or cfg.learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> float:
        """Current entropy temperature."""
        return float(self.log_alpha.exp().item())

    def _policy_actions(self, obs_vecs: np.ndarray, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            actions, _ = self.actor(self._obs_tensor(obs_vecs), deterministic)
        return actions.cpu().numpy().astype(np.float32)

    def _train_step(self) -> Dict[str, float]:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, self.device)
        alpha = self.log_alpha.exp().detach()

        critic_loss = sac_critic_loss(
            self.critic, self.critic_target, self.actor, batch, cfg.gamma, alpha
        )
        self._optimize(self.critic_optimizer, critic_loss, self.critic.parameters())

        actor_loss, log_prob = sac_actor_loss(self.critic, self.actor, batch.observations, alpha)
        self._optimize(self.actor_optimizer, actor_loss, self.actor.parameters())

        if cfg.auto_entropy:
            alpha_loss = sac_temperature_loss(self.log_alpha, log_prob, self._target_entropy)
            self._optimize(self.alpha_optimizer, alpha_loss)

        polyak_update(self.critic, self.critic_target, cfg.tau)
        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/alpha": self.alpha,
            "train/entropy": -log_prob.mean().item(),
        }
