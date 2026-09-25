"""Policy evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from relepy.core.base import BaseAgent


def evaluate_policy(
    agent: "BaseAgent",
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Run ``agent`` for ``n_episodes`` and return ``(mean_return, std_return)``."""
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + i)
        done = False
        total = 0.0
        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))
