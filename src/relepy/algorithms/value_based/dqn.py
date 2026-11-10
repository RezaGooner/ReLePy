"""Deep Q-Network (Mnih et al., 2015) with optional Double (van Hasselt et al., 2016) and
Dueling (Wang et al., 2016) extensions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Tuple

import numpy as np
import torch

from relepy.core.collector import Transition
from relepy.core.losses import dqn_loss
from relepy.core.off_policy import OffPolicyAgent, ReplayConfig
from relepy.core.policies import QNetwork
from relepy.utils.schedules import linear_schedule
from relepy.utils.spaces import check_discrete
from relepy.utils.torch_utils import polyak_update


@dataclass
class DQNConfig(ReplayConfig):
    """Hyperparameters of DQN (see :class:`~relepy.core.off_policy.ReplayConfig` for the
    replay settings).

    Attributes:
        target_update_interval: Update the target network every this many environment steps.
        tau: ``1.0`` = hard copy; ``< 1`` = soft (Polyak) update.
        epsilon_start / epsilon_end / epsilon_decay_steps: Linear epsilon-greedy schedule.
        double_dqn: Select next actions with the online net, evaluate with the target net.
        dueling: Use a dueling network architecture.
        loss: ``"huber"`` or ``"mse"``.
    """

    learning_rate: float = 1e-3
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
        if self.target_update_interval < 1:
            raise ValueError(
                f"target_update_interval must be >= 1, got {self.target_update_interval}"
            )
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        for name in ("epsilon_start", "epsilon_end"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.loss not in ("huber", "mse"):
            raise ValueError(f"loss must be 'huber' or 'mse', got {self.loss!r}")


class DQN(OffPolicyAgent):
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
    _state_attrs = ("q_net", "target_net", "optimizer")

    def _check_spaces(self) -> None:
        check_discrete(self.action_space, "action space")

    def _buffer_action_spec(self) -> Tuple[Tuple[int, ...], type]:
        return (1,), np.int64  # (B, 1) so that it can be used with ``gather``

    def _build(self) -> None:
        cfg = self.config
        self.n_actions = check_discrete(self.action_space)
        self.q_net = QNetwork(
            self._obs_dim, self.n_actions, cfg.hidden_sizes, cfg.activation, cfg.dueling
        ).to(self.device)
        self.target_net = deepcopy(self.q_net).eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self._epsilon_schedule = linear_schedule(
            cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps
        )
        self._steps_since_target_update = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon_schedule(self.num_timesteps)

    # ------------------------------------------------------------------ acting
    def _greedy_actions(self, obs_vecs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.q_net(self._obs_tensor(obs_vecs)).argmax(dim=1).cpu().numpy()

    def _select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actions = self._greedy_actions(self._preprocess_batch(obs)).astype(np.int64)
        explore = self.rng.random(self.n_envs) < self.epsilon
        if explore.any():
            actions[explore] = self.rng.integers(self.n_actions, size=int(explore.sum()))
        return actions, actions

    def predict(self, observation: Any, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(self._greedy_actions(self._preprocess(observation)[None])[0])

    # ---------------------------------------------------------------- learning
    def _train_step(self) -> Dict[str, float]:
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size, self.device)
        loss = dqn_loss(self.q_net, self.target_net, batch, cfg.gamma, cfg.double_dqn, cfg.loss)
        self._optimize(self.optimizer, loss, self.q_net.parameters())
        return {"train/loss": float(loss.item())}

    def _update_target(self) -> None:
        if self.config.tau >= 1.0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            polyak_update(self.q_net, self.target_net, self.config.tau)

    def _observe(self, transition: Transition) -> None:
        super()._observe(transition)
        self._steps_since_target_update += self.n_envs
        while self._steps_since_target_update >= self.config.target_update_interval:
            self._steps_since_target_update -= self.config.target_update_interval
            self._update_target()

    def _train_stats(self) -> Dict[str, Any]:
        return {"train/loss": self._last_stats.get("train/loss", float("nan")),
                "train/epsilon": self.epsilon}


class DoubleDQN(DQN):
    """DQN with ``double_dqn=True``."""

    _forced_config: ClassVar[Dict[str, Any]] = {"double_dqn": True}


class DuelingDQN(DQN):
    """DQN with ``dueling=True`` (combine with ``double_dqn=True`` for Double Dueling DQN)."""

    _forced_config: ClassVar[Dict[str, Any]] = {"dueling": True}
