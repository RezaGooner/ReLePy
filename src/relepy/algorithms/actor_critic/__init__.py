"""Off-policy actor-critic methods for continuous control (requires PyTorch)."""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "relepy's deep RL algorithms require PyTorch. Install with: pip install relepy[torch]"
    ) from exc

from relepy.algorithms.actor_critic.ddpg import DDPG, DDPGConfig
from relepy.algorithms.actor_critic.sac import SAC, SACConfig
from relepy.algorithms.actor_critic.td3 import TD3, TD3Config

__all__ = ["DDPG", "DDPGConfig", "SAC", "SACConfig", "TD3", "TD3Config"]
