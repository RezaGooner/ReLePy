"""Experience storage. Requires PyTorch (batches are returned as tensors)."""

from __future__ import annotations

from typing import Iterator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

Device = Union[str, torch.device]


class ReplayBatch(NamedTuple):
    observations: torch.Tensor  # (B, obs_dim)
    actions: torch.Tensor  # (B, *action_shape)
    rewards: torch.Tensor  # (B, 1)
    next_observations: torch.Tensor  # (B, obs_dim)
    dones: torch.Tensor  # (B, 1) -- 1.0 only for true termination (not time-limit truncation)


class ReplayBuffer:
    """Fixed-size FIFO buffer with uniform sampling (for off-policy algorithms).

    ``dones`` must hold *termination* flags: on a time-limit truncation the target should still
    bootstrap, so store ``terminated`` and not ``terminated or truncated``.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_shape: Tuple[int, ...] = (1,),
        action_dtype: type = np.int64,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.action_shape = tuple(action_shape)
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((capacity, *self.action_shape), dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.extend(
            np.asarray(obs)[None], np.asarray(action)[None], np.asarray([reward]),
            np.asarray(next_obs)[None], np.asarray([done]),
        )

    def extend(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Store a batch of transitions (leading dimension = batch)."""
        n = len(obs)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.observations[idx] = obs
        self.next_observations[idx] = next_obs
        self.actions[idx] = np.reshape(actions, (n, *self.action_shape))
        self.rewards[idx] = rewards
        self.dones[idx] = np.asarray(dones, dtype=np.float32)
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: Device = "cpu") -> ReplayBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[idx], device=device)

        return ReplayBatch(
            t(self.observations),
            t(self.actions),
            t(self.rewards).unsqueeze(-1),
            t(self.next_observations),
            t(self.dones).unsqueeze(-1),
        )


class RolloutBatch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Fixed-length on-policy buffer of shape ``(n_steps, n_envs, ...)`` with GAE(lambda).

    ``dones[t, i]`` marks that the episode of env ``i`` ended (terminated *or* truncated) right
    after step ``t``. For time-limit truncations the agent adds ``gamma * V(s_final)`` to the
    reward before calling :meth:`add`, which is what ReLePy's on-policy agents do.
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        action_shape: Tuple[int, ...] = (),
        action_dtype: type = np.int64,
        rng: Optional[np.random.Generator] = None,
        n_envs: int = 1,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_shape = tuple(action_shape)
        shape = (n_steps, n_envs)
        self.observations = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros((*shape, *self.action_shape), dtype=action_dtype)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.pos = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def full(self) -> bool:
        return self.pos >= self.n_steps

    @property
    def size(self) -> int:
        """Number of samples once full (``n_steps * n_envs``)."""
        return self.n_steps * self.n_envs

    def reset(self) -> None:
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
        done: Union[bool, np.ndarray],
        value: Union[float, np.ndarray],
        log_prob: Union[float, np.ndarray],
    ) -> None:
        """Add one step for every env (scalars are accepted when ``n_envs == 1``)."""
        if self.full:
            raise RuntimeError("RolloutBuffer is full; call reset() first")
        i, n = self.pos, self.n_envs
        self.observations[i] = np.reshape(obs, (n, self.obs_dim))
        self.actions[i] = np.reshape(action, (n, *self.action_shape))
        self.rewards[i] = np.reshape(reward, (n,))
        self.dones[i] = np.reshape(np.asarray(done, dtype=np.float32), (n,))
        self.values[i] = np.reshape(value, (n,))
        self.log_probs[i] = np.reshape(log_prob, (n,))
        self.pos += 1

    def compute_returns_and_advantages(
        self, last_value: Union[float, np.ndarray], gamma: float, gae_lambda: float
    ) -> None:
        """Fill ``advantages`` and ``returns`` using GAE(lambda), per environment."""
        last_values = np.broadcast_to(np.asarray(last_value, dtype=np.float32), (self.n_envs,))
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_value = last_values if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get(
        self, batch_size: Optional[int] = None, device: Device = "cpu"
    ) -> Iterator[RolloutBatch]:
        """Yield shuffled mini-batches over all ``n_steps * n_envs`` samples
        (a single full batch if ``batch_size`` is ``None``)."""
        if not self.full:
            raise RuntimeError("RolloutBuffer is not full")
        n = self.size
        size = n if batch_size is None else batch_size
        order = torch.as_tensor(self.rng.permutation(n), device=device, dtype=torch.long)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr.reshape(n, *arr.shape[2:]), device=device)

        data = (
            t(self.observations),
            t(self.actions),
            t(self.values),
            t(self.log_probs),
            t(self.advantages),
            t(self.returns),
        )
        for start in range(0, n, size):
            idx = order[start : start + size]
            yield RolloutBatch(*(x[idx] for x in data))
