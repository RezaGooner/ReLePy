"""Shared code for actor-critic style agents (REINFORCE, A2C, PPO)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces

from relepy.core.base import BaseAgent
from relepy.core.buffers import RolloutBuffer
from relepy.core.config import DeepConfig
from relepy.core.policies import ActorCriticPolicy
from relepy.utils.spaces import check_discrete, make_obs_preprocessor
from relepy.utils.torch_utils import get_device, to_cpu


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
        n_steps: Environment steps collected per update.
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
    """Owns the actor-critic network, optimizer, action conversion and persistence."""

    config: ActorCriticConfig

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
        obs_dim, self._preprocess = make_obs_preprocessor(self.observation_space)
        self._obs_dim = obs_dim
        self._discrete = isinstance(self.action_space, spaces.Discrete)
        if self._discrete:
            self._action_shape: Tuple[int, ...] = ()
            self._action_np_dtype: type = np.int64
        else:
            self._action_shape = (int(np.prod(self.action_space.shape)),)
            self._action_np_dtype = np.float32
        self.policy = ActorCriticPolicy(
            obs_dim, self.action_space, cfg.hidden_sizes, cfg.activation, cfg.log_std_init
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ------------------------------------------------------------------ acting
    def _to_env_action(self, raw: np.ndarray) -> Any:
        if self._discrete:
            return int(raw)
        low, high = self.action_space.low, self.action_space.high
        return np.clip(raw.reshape(self.action_space.shape), low, high).astype(np.float32)

    def _act(self, obs_vec: np.ndarray, deterministic: bool = False):
        """Return ``(env_action, raw_action, value, log_prob)`` for one observation."""
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, value, log_prob = self.policy.act(obs_t, deterministic)
        raw = action[0].cpu().numpy()
        return self._to_env_action(raw), raw, float(value[0]), float(log_prob[0])

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        return self._act(self._preprocess(observation), deterministic)[0]

    def _value(self, obs_vec: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.policy.value(obs_t)[0])

    def _action_tensor(self, actions: np.ndarray) -> torch.Tensor:
        dtype = torch.long if self._discrete else torch.float32
        return torch.as_tensor(actions, dtype=dtype, device=self.device)

    def _optimize(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

    # ------------------------------------------------------------- persistence
    def _get_state(self) -> Dict[str, Any]:
        return {
            "policy": to_cpu(self.policy.state_dict()),
            "optimizer": to_cpu(self.optimizer.state_dict()),
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])


class OnPolicyAgent(ActorCriticAgent):
    """Rollout collection with GAE for A2C and PPO (single environment).

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
        )

    def _collect_rollout(self) -> bool:
        """Fill the rollout buffer. Returns ``False`` if a callback requested a stop."""
        cfg = self.config
        buf = self.rollout_buffer
        buf.reset()
        while not buf.full:
            obs_vec = self._preprocess(self._obs)
            env_action, raw, value, log_prob = self._act(obs_vec)
            next_obs, reward, terminated, truncated, _ = self._env_step(env_action)
            done = terminated or truncated
            if truncated and not terminated:
                # Time-limit cut-off: the episode did not really end, so bootstrap from V(s').
                reward += cfg.gamma * self._value(self._preprocess(next_obs))
            buf.add(obs_vec, raw, reward, done, value, log_prob)
            if done:
                self._reset_env()
            else:
                self._obs = next_obs
            if not self._on_step():
                return False
        last_value = self._value(self._preprocess(self._obs))
        buf.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)
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
