"""Deep Deterministic Policy Gradient (Lillicrap et al., 2015)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from relepy.algorithms.actor_critic._base import OffPolicyAgent, OffPolicyConfig
from relepy.core.policies import ContinuousCritic, DeterministicActor
from relepy.utils.torch_utils import polyak_update, to_cpu


@dataclass
class DDPGConfig(OffPolicyConfig):
    """DDPG hyperparameters.

    The last four fields exist so that TD3 can reuse the implementation; with their DDPG
    defaults they reduce to the original algorithm.

    Attributes:
        exploration_noise: Std of Gaussian action noise (in normalized ``[-1, 1]`` units).
        target_policy_noise: Std of the target-policy smoothing noise (TD3).
        target_noise_clip: Clipping range of the smoothing noise (TD3).
        policy_delay: Update the actor and target networks every this many critic updates.
        n_critics: Number of critics; the target uses their minimum (``2`` = clipped double Q).
    """

    exploration_noise: float = 0.1
    target_policy_noise: float = 0.0
    target_noise_clip: float = 0.5
    policy_delay: int = 1
    n_critics: int = 1

    def validate(self) -> None:
        super().validate()
        if self.exploration_noise < 0 or self.target_policy_noise < 0:
            raise ValueError("noise standard deviations must be >= 0")
        if self.target_noise_clip <= 0:
            raise ValueError("target_noise_clip must be > 0")
        if self.policy_delay < 1 or self.n_critics < 1:
            raise ValueError("policy_delay and n_critics must be >= 1")


class DDPG(OffPolicyAgent):
    """DDPG for bounded continuous action spaces.

    Example:
        >>> agent = DDPG("Pendulum-v1", exploration_noise=0.1, seed=0)  # doctest: +SKIP
        >>> agent.fit(20_000)  # doctest: +SKIP
    """

    config_class = DDPGConfig
    config: DDPGConfig

    def _build(self) -> None:
        cfg = self.config
        self.actor = DeterministicActor(
            self._obs_dim, self.act_dim, cfg.hidden_sizes, cfg.activation
        ).to(self.device)
        self.critic = ContinuousCritic(
            self._obs_dim, self.act_dim, cfg.hidden_sizes, cfg.activation, cfg.n_critics
        ).to(self.device)
        self.actor_target = deepcopy(self.actor).eval()
        self.critic_target = deepcopy(self.critic).eval()
        for p in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            p.requires_grad_(False)
        actor_lr = cfg.actor_learning_rate or cfg.learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self._n_updates = 0

    def _policy_action(self, obs_vec: np.ndarray, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(self._obs_tensor(obs_vec))[0].cpu().numpy()
        if not deterministic and self.config.exploration_noise > 0:
            action = action + self.rng.normal(0.0, self.config.exploration_noise, action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _train_step(self) -> Dict[str, float]:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            next_actions = self.actor_target(batch.next_observations)
            if cfg.target_policy_noise > 0:
                noise = torch.randn_like(next_actions) * cfg.target_policy_noise
                noise = noise.clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            next_q = self.critic_target.q_min(batch.next_observations, next_actions)
            target = batch.rewards + cfg.gamma * (1.0 - batch.dones) * next_q

        critic_loss = sum(
            F.mse_loss(q, target) for q in self.critic(batch.observations, batch.actions)
        )
        self._optimize(self.critic_optimizer, critic_loss, self.critic.parameters())
        stats = {"train/critic_loss": critic_loss.item()}

        self._n_updates += 1
        if self._n_updates % cfg.policy_delay == 0:
            actor_loss = -self.critic.q1(batch.observations, self.actor(batch.observations)).mean()
            self._optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
            polyak_update(self.actor, self.actor_target, cfg.tau)
            polyak_update(self.critic, self.critic_target, cfg.tau)
            stats["train/actor_loss"] = actor_loss.item()
        return stats

    # ------------------------------------------------------------- persistence
    def _get_state(self) -> Dict[str, Any]:
        return {
            name: to_cpu(getattr(self, name).state_dict())
            for name in (
                "actor", "actor_target", "critic", "critic_target",
                "actor_optimizer", "critic_optimizer",
            )
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            getattr(self, name).load_state_dict(value)
