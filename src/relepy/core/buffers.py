"""Experience storage. Requires PyTorch (batches are returned as tensors)."""

from __future__ import annotations

from typing import Iterator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch


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
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=action_dtype)
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
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: Union[str, torch.device] = "cpu") -> ReplayBatch:
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
    """Fixed-length on-policy buffer with Generalized Advantage Estimation (GAE).

    ``dones[t]`` marks that the episode ended (terminated *or* truncated) right after step ``t``.
    For time-limit truncations the agent should add ``gamma * V(s_final)`` to the reward before
    calling :meth:`add`, which is what ReLePy's on-policy agents do.
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        action_shape: Tuple[int, ...] = (),
        action_dtype: type = np.int64,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_steps = n_steps
        self.observations = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.pos = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def full(self) -> bool:
        return self.pos >= self.n_steps

    def reset(self) -> None:
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full; call reset() first")
        i = self.pos
        self.observations[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.pos += 1

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        """Fill ``advantages`` and ``returns`` using GAE(lambda)."""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get(
        self, batch_size: Optional[int] = None, device: Union[str, torch.device] = "cpu"
    ) -> Iterator[RolloutBatch]:
        """Yield shuffled mini-batches (a single full batch if ``batch_size`` is ``None``)."""
        if not self.full:
            raise RuntimeError("RolloutBuffer is not full")
        n = self.n_steps
        size = n if batch_size is None else batch_size
        order = torch.as_tensor(self.rng.permutation(n), device=device, dtype=torch.long)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr, device=device)

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
