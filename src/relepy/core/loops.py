"""Reusable training-loop templates (no PyTorch needed).

:class:`StepwiseAgent` is the loop of every agent that acts and learns one (vector) step at
a time: tabular TD control, DQN and the off-policy actor-critics. A subclass only decides
*what to do* (choose actions, learn from a transition); the template handles *when*.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Tuple

import numpy as np

from relepy.core.base import BaseAgent
from relepy.core.collector import Transition


class StepwiseAgent(BaseAgent):
    """``act -> step -> observe`` loop. ``fit(..., log_interval=k)`` logs every ``k`` episodes.

    Subclasses implement:

    * :meth:`_select_actions` -- actions for the current batch of observations,
    * :meth:`_observe` -- learn from / store the resulting :class:`Transition`,
    * optionally :meth:`_train_stats` -- values added to each training log row.
    """

    @abc.abstractmethod
    def _select_actions(self, obs: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Return ``(env_actions, recorded_actions)`` for a batch of raw observations.

        ``recorded_actions`` is what ends up in the transition (e.g. a normalized action).
        """

    @abc.abstractmethod
    def _observe(self, transition: Transition) -> None:
        """Learn from or store one vector step."""

    def _train_stats(self) -> Dict[str, Any]:
        return {}

    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        end = self.num_timesteps + total_timesteps
        collector = self.collector
        while self.num_timesteps < end:
            env_actions, actions = self._select_actions(collector.ensure_started())
            transition = collector.step(env_actions, actions)
            self._observe(transition)
            if not self._on_step():
                return
            self._log_every_episodes(log_interval, self._train_stats)
