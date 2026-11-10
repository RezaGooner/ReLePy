"""Shared machinery for tabular TD control with an epsilon-greedy behaviour policy."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from relepy.core.collector import Transition
from relepy.core.config import BaseConfig
from relepy.core.loops import StepwiseAgent
from relepy.utils.schedules import linear_schedule
from relepy.utils.spaces import check_discrete


@dataclass
class TabularConfig(BaseConfig):
    """Hyperparameters of tabular TD control.

    Attributes:
        alpha: Learning rate (step size).
        epsilon_start / epsilon_end: Exploration rate, decayed linearly.
        epsilon_decay_steps: Number of timesteps over which epsilon decays.
        initial_q_value: Initial value of every Q entry (optimistic init encourages exploration).
    """

    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    initial_q_value: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        for name in ("epsilon_start", "epsilon_end"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.epsilon_decay_steps < 0:
            raise ValueError("epsilon_decay_steps must be >= 0")


class TabularAgent(StepwiseAgent):
    """Base class for Q-table agents. Requires ``Discrete`` observation and action spaces.

    ``fit(..., log_interval=k)`` logs every ``k`` episodes. The learned table is ``q_table``
    with shape ``(n_states, n_actions)``. Works with ``n_envs > 1``: every environment's
    transition is applied to the shared table.
    """

    config_class = TabularConfig
    config: TabularConfig
    _state_attrs = ("q_table",)

    def _check_spaces(self) -> None:
        check_discrete(self.observation_space, "observation space")
        check_discrete(self.action_space, "action space")

    def _setup(self) -> None:
        self.n_states = check_discrete(self.observation_space)
        self.n_actions = check_discrete(self.action_space)
        self.q_table = np.full(
            (self.n_states, self.n_actions), self.config.initial_q_value, dtype=np.float64
        )
        self._epsilon_schedule = linear_schedule(
            self.config.epsilon_start, self.config.epsilon_end, self.config.epsilon_decay_steps
        )
        # SARSA must execute the next action it bootstrapped from: -1 = none chosen yet
        self._pending_actions = np.full(self.n_envs, -1, dtype=np.int64)

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon_schedule(self.num_timesteps)

    # ------------------------------------------------------------------ policy
    def _greedy_action(self, state: int, random_ties: bool = True) -> int:
        q = self.q_table[state]
        if not random_ties:
            return int(np.argmax(q))
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def _explore(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action(state)

    def predict(self, observation: Any, deterministic: bool = True) -> int:
        state = int(observation)
        if deterministic:
            return self._greedy_action(state, random_ties=False)
        return self._explore(state)

    # ---------------------------------------------------------------- learning
    @abc.abstractmethod
    def _td_target(
        self, reward: float, next_state: int, next_action: int, terminated: bool
    ) -> float:
        """Return the bootstrapped target for the update."""

    def _select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.empty(self.n_envs, dtype=np.int64)
        for i in range(self.n_envs):
            pending = self._pending_actions[i]
            actions[i] = pending if pending >= 0 else self._explore(int(obs[i]))
        self._pending_actions[:] = -1
        return actions, actions

    def _observe(self, transition: Transition) -> None:
        alpha = self.config.alpha
        for i in range(self.n_envs):
            state, action = int(transition.obs[i]), int(transition.actions[i])
            next_state = int(transition.next_obs[i])
            terminated = bool(transition.terminated[i])
            next_action = self._explore(next_state)
            target = self._td_target(float(transition.rewards[i]), next_state, next_action,
                                     terminated)
            self.q_table[state, action] += alpha * (target - self.q_table[state, action])
            if not (terminated or transition.truncated[i]):
                self._pending_actions[i] = next_action

    def _train_stats(self) -> Dict[str, Any]:
        return {"train/epsilon": self.epsilon}
