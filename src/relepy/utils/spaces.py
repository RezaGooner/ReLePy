"""Helpers for working with Gymnasium environments and spaces."""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def make_env(env: Union[str, gym.Env], **kwargs: Any) -> gym.Env:
    """Return a Gymnasium environment from an id string or pass an existing one through."""
    if isinstance(env, str):
        return gym.make(env, **kwargs)
    if isinstance(env, gym.Env):
        return env
    raise TypeError(f"env must be a Gymnasium id or gymnasium.Env, got {type(env).__name__}")


def check_discrete(space: spaces.Space, name: str = "space") -> int:
    """Validate a ``Discrete`` space with ``start == 0`` and return its size."""
    if not isinstance(space, spaces.Discrete):
        raise TypeError(f"{name} must be gymnasium.spaces.Discrete, got {type(space).__name__}")
    if int(space.start) != 0:
        raise TypeError(f"{name} must start at 0, got start={int(space.start)}")
    return int(space.n)


def make_obs_preprocessor(space: spaces.Space) -> Tuple[int, Callable[[Any], np.ndarray]]:
    """Return ``(feature_dim, fn)`` where ``fn`` maps one raw observation to a float32 vector.

    ``Discrete`` observations are one-hot encoded, ``Box`` observations are flattened.
    """
    if isinstance(space, spaces.Discrete):
        n = check_discrete(space, "observation space")

        def one_hot(obs: Any) -> np.ndarray:
            vec = np.zeros(n, dtype=np.float32)
            vec[int(obs)] = 1.0
            return vec

        return n, one_hot

    if isinstance(space, spaces.Box):
        dim = int(np.prod(space.shape))

        def flatten(obs: Any) -> np.ndarray:
            return np.asarray(obs, dtype=np.float32).reshape(-1)

        return dim, flatten

    raise TypeError(
        f"Unsupported observation space {type(space).__name__}; use Discrete or Box "
        "(wrap the environment to convert other spaces)."
    )
