"""Templates for actor-critic policy-gradient agents: REINFORCE, A2C, PPO.

Requires PyTorch.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import torch
from gymnasium import spaces

from relepy.core.base import BaseAgent
from relepy.core.buffers import RolloutBuffer
from relepy.core.config import DeepConfig
from relepy.core.policies import ActorCriticPolicy
from relepy.utils.spaces import check_discrete, make_batch_preprocessor, make_obs_preprocessor
from relepy.utils.torch_utils import get_device


@dataclass
class ActorCriticConfig(DeepConfig):
    """Attributes shared by policy-gradient methods.

    Attributes:
        entropy_coef: Weight of the entropy bonus (encourages exploration).
        value_coef: Weight of the value-function loss.
        normalize_advantage: Standardize advantages before the policy update.
        log_std_init: Initial log standard deviation (continuous actions only).
    """

    activation: str = "tanh"
    max_grad_norm: Optional[float] = 0.5
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    normalize_advantage: bool = True
    log_std_init: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.entropy_coef < 0 or self.value_coef < 0:
            raise ValueError("entropy_coef and value_coef must be >= 0")


@dataclass
class OnPolicyConfig(ActorCriticConfig):
    """Adds rollout settings.

    Attributes:
        n_steps: Steps collected *per environment* per update (``n_steps * n_envs`` samples).
        gae_lambda: GAE lambda (``1.0`` = Monte-Carlo advantages, ``0`` = one-step TD).
    """

    n_steps: int = 2048
    gae_lambda: float = 0.95

    def validate(self) -> None:
        super().validate()
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")


class ActorCriticAgent(BaseAgent):
    """Owns the actor-critic network, the optimizer and action conversion."""

    config: ActorCriticConfig
    _state_attrs = ("policy", "optimizer")

    def _check_spaces(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            check_discrete(self.action_space, "action space")
        elif not isinstance(self.action_space, spaces.Box):
            raise TypeError(
                f"Unsupported action space {type(self.action_space).__name__}; "
                "use Discrete or Box."
            )

    def _setup(self) -> None:
        cfg = self.config
        self.device = get_device(cfg.device)
        self._obs_dim, self._preprocess = make_obs_preprocessor(self.observation_space)
        _, self._preprocess_batch = make_batch_preprocessor(self.observation_space)
        self._discrete = isinstance(self.action_space, spaces.Discrete)
        if self._discrete:
            self._action_shape: Tuple[int, ...] = ()
            self._action_np_dtype: type = np.int64
        else:
            box = cast(spaces.Box, self.action_space)
            self._action_shape = (int(np.prod(box.shape)),)
            self._action_np_dtype = np.float32
        self.policy = ActorCriticPolicy(
            self._obs_dim, self.action_space, cfg.hidden_sizes, cfg.activation, cfg.log_std_init
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ------------------------------------------------------------------ acting
    def _to_env_actions(self, raw: np.ndarray) -> Any:
        """Batch of raw policy outputs -> actions in the environment's format."""
        if self._discrete:
            return raw.astype(np.int64)
        box = cast(spaces.Box, self.action_space)
        shaped = raw.reshape((raw.shape[0], *box.shape))
        return np.clip(shaped, box.low, box.high).astype(np.float32)

    def _act(
        self, obs_vecs: np.ndarray, deterministic: bool = False
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(env_actions, raw_actions, values, log_probs)`` for a batch."""
        obs_t = torch.as_tensor(obs_vecs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, value, log_prob = self.policy.act(obs_t, deterministic)
        raw = action.cpu().numpy()
        return self._to_env_actions(raw), raw, value.cpu().numpy(), log_prob.cpu().numpy()

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        env_actions = self._act(self._preprocess(observation)[None], deterministic)[0]
        return int(env_actions[0]) if self._discrete else env_actions[0]

    def _values(self, obs_vecs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_vecs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.policy.value(obs_t).cpu().numpy()

    def _action_tensor(self, actions: np.ndarray) -> torch.Tensor:
        dtype = torch.long if self._discrete else torch.float32
        return torch.as_tensor(actions, dtype=dtype, device=self.device)

    def _optimize(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()


class OnPolicyAgent(ActorCriticAgent):
    """Collects ``n_steps`` per environment, computes GAE, then calls :meth:`_update`.

    ``fit(..., log_interval=k)`` logs every ``k`` updates.
    """

    config: OnPolicyConfig

    def _setup(self) -> None:
        super()._setup()
        self.rollout_buffer = RolloutBuffer(
            self.config.n_steps,
            self._obs_dim,
            self._action_shape,
            self._action_np_dtype,
            rng=self.rng,
            n_envs=self.n_envs,
        )

    def _collect_rollout(self) -> bool:
        """Fill the rollout buffer. Returns ``False`` if a callback requested a stop."""
        cfg = self.config
        buf = self.rollout_buffer
        buf.reset()
        while not buf.full:
            obs_vecs = self._preprocess_batch(self.collector.ensure_started())
            env_actions, raw, values, log_probs = self._act(obs_vecs)
            tr = self.collector.step(env_actions, raw)
            rewards = tr.rewards.copy()
            cut = np.logical_and(tr.truncated, np.logical_not(tr.terminated))
            if cut.any():
                # Time-limit cut-off: the episode did not really end, so bootstrap from V(s').
                final = self._preprocess_batch(tr.next_obs[cut])
                rewards[cut] += cfg.gamma * self._values(final)
            buf.add(obs_vecs, raw, rewards, tr.dones, values, log_probs)
            if not self._on_step():
                return False
        last_values = self._values(self._preprocess_batch(self.collector.obs))
        buf.compute_returns_and_advantages(last_values, cfg.gamma, cfg.gae_lambda)
        return True

    @abc.abstractmethod
    def _update(self) -> Dict[str, float]:
        """Update the policy from the filled rollout buffer; return stats to log."""

    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        end = self.num_timesteps + total_timesteps
        n_updates = 0
        while self.num_timesteps < end:
            if not self._collect_rollout():
                return
            stats = self._update()
            n_updates += 1
            if n_updates % log_interval == 0:
                self._log_training(stats)
