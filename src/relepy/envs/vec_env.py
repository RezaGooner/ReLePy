"""A small synchronous vector environment with *same-step* auto-reset.

ReLePy uses its own vector env instead of ``gymnasium.vector`` because Gymnasium changed its
auto-reset semantics between versions (0.29 resets in the same step, 1.0 in the *next* step).
:class:`SyncVecEnv` always behaves the same way:

* ``step`` returns the observation *after* auto-reset for finished sub-environments, and
* the true last observation of a finished episode is returned separately in ``final_obs``
  (needed to bootstrap value targets on time-limit truncation).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Union

import gymnasium as gym
import numpy as np

EnvFactory = Callable[[], gym.Env]
EnvLike = Union[str, gym.Env, EnvFactory]


class VecStep(NamedTuple):
    """Result of :meth:`SyncVecEnv.step`. Arrays have a leading ``n_envs`` dimension."""

    obs: np.ndarray  # observation to act on next (already reset for finished envs)
    rewards: np.ndarray  # (n_envs,) float32
    terminated: np.ndarray  # (n_envs,) bool
    truncated: np.ndarray  # (n_envs,) bool
    final_obs: np.ndarray  # true next observation (== obs where the episode did not end)
    infos: List[Dict[str, Any]]


def env_factory(env: EnvLike, **make_kwargs: Any) -> Optional[EnvFactory]:
    """Return a factory creating fresh copies of ``env``, or ``None`` for an env instance."""
    if isinstance(env, str):
        return lambda: gym.make(env, **make_kwargs)
    if isinstance(env, gym.Env):
        return None
    if callable(env):
        return env
    raise TypeError(
        f"env must be a Gymnasium id, a gymnasium.Env or a factory callable, "
        f"got {type(env).__name__}"
    )


class SyncVecEnv:
    """Steps ``n`` Gymnasium environments sequentially in the same process.

    Args:
        envs: The environments. They must share observation and action spaces.
    """

    def __init__(self, envs: Sequence[gym.Env]) -> None:
        if len(envs) == 0:
            raise ValueError("SyncVecEnv needs at least one environment")
        self.envs: List[gym.Env] = list(envs)
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        for e in self.envs[1:]:
            if e.observation_space != self.observation_space or e.action_space != self.action_space:
                raise ValueError("All environments of a SyncVecEnv must share their spaces")

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset all envs. With a seed, env ``i`` is seeded with ``seed + i``."""
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=None if seed is None else seed + i)
            if seed is not None:
                env.action_space.seed(seed + i)
            obs.append(o)
        return np.stack([np.asarray(o) for o in obs])

    def step(self, actions: Union[np.ndarray, Sequence[Any]]) -> VecStep:
        if len(actions) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} actions, got {len(actions)}")
        obs, final_obs, infos = [], [], []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = env.step(action)
            rewards[i], terminated[i], truncated[i] = float(r), bool(term), bool(trunc)
            final_obs.append(np.asarray(o))
            if term or trunc:
                o, _ = env.reset()
            obs.append(np.asarray(o))
            infos.append(info)
        return VecStep(np.stack(obs), rewards, terminated, truncated, np.stack(final_obs), infos)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __len__(self) -> int:
        return self.n_envs

    def __repr__(self) -> str:
        return f"SyncVecEnv(n_envs={self.n_envs}, env={self.envs[0]})"


def make_vec_env(env: Union[EnvLike, SyncVecEnv], n_envs: int = 1) -> SyncVecEnv:
    """Build a :class:`SyncVecEnv` from an id, a factory, an env instance or a vec env.

    An env *instance* can only be used with ``n_envs=1`` (it cannot be copied safely); pass an
    id or a factory such as ``lambda: gym.make("CartPole-v1")`` for several environments.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    if isinstance(env, SyncVecEnv):
        if env.n_envs != n_envs:
            raise ValueError(f"Got a SyncVecEnv with {env.n_envs} envs but n_envs={n_envs}")
        return env
    factory = env_factory(env)
    if factory is None:
        if n_envs != 1:
            raise ValueError(
                "n_envs > 1 needs an env id or a factory callable, not an env instance"
            )
        return SyncVecEnv([env])  # type: ignore[list-item]
    return SyncVecEnv([factory() for _ in range(n_envs)])
