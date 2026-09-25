"""Soft Actor-Critic (Haarnoja et al., 2018) with automatic entropy tuning."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from relepy.algorithms.actor_critic._base import OffPolicyAgent, OffPolicyConfig
from relepy.core.policies import ContinuousCritic, SquashedGaussianActor
from relepy.utils.torch_utils import polyak_update, to_cpu


@dataclass
class SACConfig(OffPolicyConfig):
    """SAC hyperparameters.

    Attributes:
        auto_entropy: Learn the entropy temperature alpha to match ``target_entropy``.
        initial_alpha: Initial (or, if ``auto_entropy=False``, fixed) temperature.
        target_entropy: Desired policy entropy; ``None`` uses ``-action_dim``.
    """

    auto_entropy: bool = True
    initial_alpha: float = 1.0
    target_entropy: Optional[float] = None

    def validate(self) -> None:
        super().validate()
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha must be > 0")


class SAC(OffPolicyAgent):
    """SAC for bounded continuous action spaces.

    Example:
        >>> agent = SAC("Pendulum-v1", seed=0)  # doctest: +SKIP
        >>> agent.fit(15_000)  # doctest: +SKIP
    """

    config_class = SACConfig
    config: SACConfig

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
        actor_lr = cfg.actor_learning_rate or cfg.learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)

    @property
    def alpha(self) -> float:
        """Current entropy temperature."""
        return float(self.log_alpha.exp().item())

    def _policy_action(self, obs_vec: np.ndarray, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actor(self._obs_tensor(obs_vec), deterministic)
        return action[0].cpu().numpy().astype(np.float32)

    def _train_step(self) -> Dict[str, float]:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, self.device)
        alpha = self.log_alpha.exp().detach()

        # critic update
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(batch.next_observations)
            next_q = self.critic_target.q_min(batch.next_observations, next_actions)
            next_q = next_q - alpha * next_log_prob.unsqueeze(1)
            target = batch.rewards + cfg.gamma * (1.0 - batch.dones) * next_q
        critic_loss = sum(
            F.mse_loss(q, target) for q in self.critic(batch.observations, batch.actions)
        )
        self._optimize(self.critic_optimizer, critic_loss, self.critic.parameters())

        # actor update
        actions, log_prob = self.actor(batch.observations)
        q_new = self.critic.q_min(batch.observations, actions)
        actor_loss = (alpha * log_prob.unsqueeze(1) - q_new).mean()
        self._optimize(self.actor_optimizer, actor_loss, self.actor.parameters())

        # temperature update
        if cfg.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self._target_entropy)).mean()
            self._optimize(self.alpha_optimizer, alpha_loss)

        polyak_update(self.critic, self.critic_target, cfg.tau)
        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/alpha": self.alpha,
            "train/entropy": -log_prob.mean().item(),
        }

    # ------------------------------------------------------------- persistence
    _MODULES = ("actor", "critic", "critic_target")
    _OPTIMIZERS = ("actor_optimizer", "critic_optimizer", "alpha_optimizer")

    def _get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            name: to_cpu(getattr(self, name).state_dict())
            for name in self._MODULES + self._OPTIMIZERS
        }
        state["log_alpha"] = self.log_alpha.detach().cpu()
        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        for name in self._MODULES + self._OPTIMIZERS:
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
