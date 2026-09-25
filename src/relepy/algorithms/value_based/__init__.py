"""Value-based deep RL (requires PyTorch)."""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "relepy's deep RL algorithms require PyTorch. Install with: pip install relepy[torch]"
    ) from exc

from relepy.algorithms.value_based.dqn import DQN, DoubleDQN, DQNConfig, DuelingDQN

__all__ = ["DQN", "DQNConfig", "DoubleDQN", "DuelingDQN"]
