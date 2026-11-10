"""The :class:`Collector` is the only component that talks to the environment.

It owns a :class:`~relepy.envs.SyncVecEnv`, keeps the current observations, counts timesteps
and episodes and records episode statistics. Agents never call ``env.step`` themselves, so
every algorithm gets vectorized collection, seeding and statistics for free.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional

import numpy as np

from relepy.envs.vec_env import SyncVecEnv


class Transition(NamedTuple):
    """One vector step. Every array has a leading ``n_envs`` dimension.

    ``next_obs`` is the *true* next observation (the final one for finished episodes), so it is
    what value targets must bootstrap from. ``terminated`` marks a real end of the episode;
    ``truncated`` a time-limit cut-off, after which the target should still bootstrap.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: List[Dict[str, Any]]

    @property
    def dones(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


class Collector:
    """Steps a vector environment and tracks progress.

    Args:
        vec_env: The environments to collect from.
        seed: Seed used for the very first reset (``seed + i`` for env ``i``).
        stats_window: Number of recent episodes kept in :attr:`episode_rewards`.
    """

    def __init__(
        self, vec_env: SyncVecEnv, seed: Optional[int] = None, stats_window: int = 100
    ) -> None:
        self.vec_env = vec_env
        self.n_envs = vec_env.n_envs
        self.seed = seed
        self.obs: Optional[np.ndarray] = None
        self.num_timesteps = 0
        self.n_episodes = 0
        self.episode_rewards: Deque[float] = deque(maxlen=stats_window)
        self.episode_lengths: Deque[int] = deque(maxlen=stats_window)
        self._ep_returns = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(self.n_envs, dtype=np.int64)
        self._seeded = False

    def reset(self) -> np.ndarray:
        """Reset every environment (seeded on the first call) and return the observations."""
        seed = None if self._seeded else self.seed
        self._seeded = True
        self.obs = self.vec_env.reset(seed=seed)
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        return self.obs

    def ensure_started(self) -> np.ndarray:
        return self.reset() if self.obs is None else self.obs

    def step(self, env_actions: Any, actions: Optional[np.ndarray] = None) -> Transition:
        """Apply one action per environment.

        Args:
            env_actions: Actions in the environment's format.
            actions: What to record in the transition (e.g. normalized actions); defaults to
                ``env_actions``.
        """
        obs = self.ensure_started()
        out = self.vec_env.step(env_actions)
        self.num_timesteps += self.n_envs
        self._ep_returns += out.rewards
        self._ep_lengths += 1
        dones = np.logical_or(out.terminated, out.truncated)
        for i in np.flatnonzero(dones):
            self.episode_rewards.append(float(self._ep_returns[i]))
            self.episode_lengths.append(int(self._ep_lengths[i]))
            self.n_episodes += 1
            self._ep_returns[i] = 0.0
            self._ep_lengths[i] = 0
        self.obs = out.obs
        recorded = np.asarray(env_actions) if actions is None else np.asarray(actions)
        return Transition(
            obs, recorded, out.rewards, out.final_obs, out.terminated, out.truncated, out.infos
        )
