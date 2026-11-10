"""Deep Deterministic Policy Gradient (Lillicrap et al., 2015)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from relepy.algorithms.actor_critic._base import ContinuousOffPolicyAgent, OffPolicyConfig
from relepy.core.losses import deterministic_actor_loss, deterministic_critic_loss
from relepy.core.policies import ContinuousCritic, DeterministicActor
from relepy.utils.torch_utils import polyak_update


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


class DDPG(ContinuousOffPolicyAgent):
    """DDPG for bounded continuous action spaces.

    Example:
        >>> agent = DDPG("Pendulum-v1", exploration_noise=0.1, seed=0)  # doctest: +SKIP
        >>> agent.fit(20_000)  # doctest: +SKIP
    """

    config_class = DDPGConfig
    config: DDPGConfig
    _state_attrs = (
        "actor", "actor_target", "critic", "critic_target", "actor_optimizer", "critic_optimizer",
    )

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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self._n_updates = 0

    def _policy_actions(self, obs_vecs: np.ndarray, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            actions = self.actor(self._obs_tensor(obs_vecs)).cpu().numpy()
        if not deterministic and self.config.exploration_noise > 0:
            actions = actions + self.rng.normal(0.0, self.config.exploration_noise, actions.shape)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def _train_step(self) -> Dict[str, float]:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, self.device)
        critic_loss = deterministic_critic_loss(
            self.critic, self.critic_target, self.actor_target, batch, cfg.gamma,
            cfg.target_policy_noise, cfg.target_noise_clip,
        )
        self._optimize(self.critic_optimizer, critic_loss, self.critic.parameters())
        stats = {"train/critic_loss": critic_loss.item()}

        self._n_updates += 1
        if self._n_updates % cfg.policy_delay == 0:
            actor_loss = deterministic_actor_loss(self.critic, self.actor, batch.observations)
            self._optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
            polyak_update(self.actor, self.actor_target, cfg.tau)
            polyak_update(self.critic, self.critic_target, cfg.tau)
            stats["train/actor_loss"] = actor_loss.item()
        return stats
