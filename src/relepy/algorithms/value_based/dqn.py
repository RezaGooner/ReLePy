"""Deep Q-Network (Mnih et al., 2015) with optional Double (van Hasselt et al., 2016) and
Dueling (Wang et al., 2016) extensions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

import numpy as np
import torch
import torch.nn.functional as F

from relepy.core.base import BaseAgent
from relepy.core.buffers import ReplayBuffer
from relepy.core.config import DeepConfig
from relepy.core.policies import QNetwork
from relepy.utils.schedules import linear_schedule
from relepy.utils.spaces import check_discrete, make_obs_preprocessor
from relepy.utils.torch_utils import get_device, polyak_update, to_cpu


@dataclass
class DQNConfig(DeepConfig):
    """Hyperparameters of DQN.

    Attributes:
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size per gradient step.
        learning_starts: Collect this many steps with the initial policy before learning.
        train_freq: Do ``gradient_steps`` updates every ``train_freq`` environment steps.
        gradient_steps: Gradient steps per training phase.
        target_update_interval: Update the target network every this many steps.
        tau: ``1.0`` = hard copy; ``< 1`` = soft (Polyak) update.
        epsilon_start / epsilon_end / epsilon_decay_steps: Linear epsilon-greedy schedule.
        double_dqn: Select next actions with the online net, evaluate with the target net.
        dueling: Use a dueling network architecture.
        loss: ``"huber"`` or ``"mse"``.
    """

    learning_rate: float = 1e-3
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    tau: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    double_dqn: bool = False
    dueling: bool = False
    loss: str = "huber"

    def validate(self) -> None:
        super().validate()
        for name in ("buffer_size", "batch_size", "train_freq", "gradient_steps",
                     "target_update_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        for name in ("epsilon_start", "epsilon_end"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.loss not in ("huber", "mse"):
            raise ValueError(f"loss must be 'huber' or 'mse', got {self.loss!r}")


class DQN(BaseAgent):
    """DQN for discrete action spaces and ``Box``/``Discrete`` observations.

    ``fit(..., log_interval=k)`` logs every ``k`` episodes. The replay buffer is not saved by
    :meth:`save`.

    Example:
        >>> agent = DQN("CartPole-v1", learning_rate=1e-3, seed=0)  # doctest: +SKIP
        >>> agent.fit(30_000)  # doctest: +SKIP
        >>> action = agent.predict(obs)  # doctest: +SKIP
    """

    config_class = DQNConfig
    config: DQNConfig

    def _check_spaces(self) -> None:
        check_discrete(self.action_space, "action space")

    def _setup(self) -> None:
        cfg = self.config
        self.device = get_device(cfg.device)
        self.n_actions = check_discrete(self.action_space)
        obs_dim, self._preprocess = make_obs_preprocessor(self.observation_space)
        self.q_net = QNetwork(
            obs_dim, self.n_actions, cfg.hidden_sizes, cfg.activation, cfg.dueling
        ).to(self.device)
        self.target_net = deepcopy(self.q_net).eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, rng=self.rng)
        self._epsilon_schedule = linear_schedule(
            cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps
        )
        self._last_loss = float("nan")

    @property
    def epsilon(self) -> float:
        return self._epsilon_schedule(self.num_timesteps)

    # ------------------------------------------------------------------ acting
    def _greedy_action(self, obs_vec: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)
            return int(self.q_net(obs_t).argmax(dim=1).item())

    def predict(self, observation: Any, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action(self._preprocess(observation))

    # ---------------------------------------------------------------- learning
    def _train_step(self) -> float:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            if cfg.double_dqn:
                best = self.q_net(batch.next_observations).argmax(dim=1, keepdim=True)
                next_q = self.target_net(batch.next_observations).gather(1, best)
            else:
                next_q = self.target_net(batch.next_observations).max(dim=1, keepdim=True)[0]
            target = batch.rewards + cfg.gamma * (1.0 - batch.dones) * next_q
        current = self.q_net(batch.observations).gather(1, batch.actions)
        loss_fn = F.smooth_l1_loss if cfg.loss == "huber" else F.mse_loss
        loss = loss_fn(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _update_target(self) -> None:
        if self.config.tau >= 1.0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            polyak_update(self.q_net, self.target_net, self.config.tau)

    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        cfg = self.config
        end = self.num_timesteps + total_timesteps
        while self.num_timesteps < end:
            obs_vec = self._preprocess(self._obs)
            if self.rng.random() < self.epsilon:
                action = int(self.rng.integers(self.n_actions))
            else:
                action = self._greedy_action(obs_vec)

            next_obs, reward, terminated, truncated, _ = self._env_step(action)
            self.buffer.add(obs_vec, action, reward, self._preprocess(next_obs), terminated)
            done = terminated or truncated
            if done:
                self._reset_env()
            else:
                self._obs = next_obs

            if (
                self.num_timesteps >= cfg.learning_starts
                and self.num_timesteps % cfg.train_freq == 0
                and len(self.buffer) >= cfg.batch_size
            ):
                for _ in range(cfg.gradient_steps):
                    self._last_loss = self._train_step()
            if self.num_timesteps % cfg.target_update_interval == 0:
                self._update_target()

            if not self._on_step():
                return
            if done and self._n_episodes % log_interval == 0:
                self._log_training(
                    {"train/loss": self._last_loss, "train/epsilon": self.epsilon}
                )

    # ------------------------------------------------------------- persistence
    def _get_state(self) -> Dict[str, Any]:
        return {
            "q_net": to_cpu(self.q_net.state_dict()),
            "target_net": to_cpu(self.target_net.state_dict()),
            "optimizer": to_cpu(self.optimizer.state_dict()),
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])


class DoubleDQN(DQN):
    """DQN with ``double_dqn=True``."""

    _forced_config: ClassVar[Dict[str, Any]] = {"double_dqn": True}


class DuelingDQN(DQN):
    """DQN with ``dueling=True`` (combine with ``double_dqn=True`` for Double Dueling DQN)."""

    _forced_config: ClassVar[Dict[str, Any]] = {"dueling": True}
