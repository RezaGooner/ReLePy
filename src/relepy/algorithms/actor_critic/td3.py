"""Twin Delayed DDPG (Fujimoto et al., 2018)."""

from __future__ import annotations

from dataclasses import dataclass

from relepy.algorithms.actor_critic.ddpg import DDPG, DDPGConfig


@dataclass
class TD3Config(DDPGConfig):
    """TD3 = DDPG with twin critics, target policy smoothing and delayed policy updates."""

    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    n_critics: int = 2


class TD3(DDPG):
    """TD3 for bounded continuous action spaces.

    Example:
        >>> agent = TD3("Pendulum-v1", seed=0)  # doctest: +SKIP
        >>> agent.fit(20_000)  # doctest: +SKIP
    """

    config_class = TD3Config
    config: TD3Config
