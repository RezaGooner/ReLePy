"""A tiny deterministic grid world, handy for demos and tests."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """``size x size`` grid. The agent starts at the top-left and must reach the bottom-right.

    * Observation: ``Discrete(size * size)`` (``row * size + col``).
    * Actions: ``Discrete(4)`` -- 0 up, 1 right, 2 down, 3 left. Walls block movement.
    * Reward: ``+1`` on reaching the goal, ``step_penalty`` on every other step.
    * Episode ends on reaching the goal (terminated) or after ``max_steps`` (truncated).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self, size: int = 4, max_steps: int = 50, step_penalty: float = -0.01
    ) -> None:
        super().__init__()
        if size < 2:
            raise ValueError("size must be >= 2")
        self.size = size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self._pos = (0, 0)
        self._steps = 0

    def _obs(self) -> int:
        return self._pos[0] * self.size + self._pos[1]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self._pos = (0, 0)
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[int(action)]
        row = int(np.clip(self._pos[0] + dr, 0, self.size - 1))
        col = int(np.clip(self._pos[1] + dc, 0, self.size - 1))
        self._pos = (row, col)
        self._steps += 1
        terminated = self._pos == (self.size - 1, self.size - 1)
        truncated = (not terminated) and self._steps >= self.max_steps
        reward = 1.0 if terminated else self.step_penalty
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> str:
        rows = []
        for r in range(self.size):
            rows.append(
                "".join(
                    "A" if (r, c) == self._pos else "G" if (r, c) == (self.size - 1, self.size - 1)
                    else "."
                    for c in range(self.size)
                )
            )
        return "\n".join(rows)
