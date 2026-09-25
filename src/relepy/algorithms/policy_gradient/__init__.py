"""Policy-gradient deep RL (requires PyTorch)."""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "relepy's deep RL algorithms require PyTorch. Install with: pip install relepy[torch]"
    ) from exc

from relepy.algorithms.policy_gradient.a2c import A2C, A2CConfig
from relepy.algorithms.policy_gradient.ppo import PPO, PPOConfig
from relepy.algorithms.policy_gradient.reinforce import REINFORCE, REINFORCEConfig

__all__ = ["A2C", "A2CConfig", "PPO", "PPOConfig", "REINFORCE", "REINFORCEConfig"]
