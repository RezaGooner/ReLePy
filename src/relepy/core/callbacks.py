"""Callbacks hook into ``agent.fit`` to evaluate, checkpoint or stop training early.

Return ``False`` from :meth:`Callback.on_step` to stop training.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

from relepy.utils.evaluation import evaluate_policy
from relepy.utils.spaces import make_env

if TYPE_CHECKING:  # pragma: no cover
    from relepy.core.base import BaseAgent


class Callback:
    """Base class. Override any of the hooks."""

    def on_training_start(self, agent: "BaseAgent") -> None:
        pass

    def on_step(self, agent: "BaseAgent") -> bool:
        """Called after every environment step. Return ``False`` to stop training."""
        return True

    def on_training_end(self, agent: "BaseAgent") -> None:
        pass


class CallbackList(Callback):
    """Runs several callbacks; training stops if any of them returns ``False``."""

    def __init__(
        self, callbacks: Optional[Union[Callback, Sequence[Callback]]] = None
    ) -> None:
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callbacks: List[Callback] = list(callbacks)

    def on_training_start(self, agent: "BaseAgent") -> None:
        for cb in self.callbacks:
            cb.on_training_start(agent)

    def on_step(self, agent: "BaseAgent") -> bool:
        keep_going = True
        for cb in self.callbacks:
            if cb.on_step(agent) is False:
                keep_going = False
        return keep_going

    def on_training_end(self, agent: "BaseAgent") -> None:
        for cb in self.callbacks:
            cb.on_training_end(agent)


class EvalCallback(Callback):
    """Periodically evaluate the agent on a separate environment.

    Args:
        eval_env: Environment (or Gymnasium id) used for evaluation.
        eval_freq: Evaluate every ``eval_freq`` timesteps.
        n_eval_episodes: Episodes per evaluation.
        deterministic: Use the deterministic policy.
        best_model_path: If given, save the best agent here.
        reward_threshold: If given, stop training once the mean eval return reaches it.
    """

    def __init__(
        self,
        eval_env: Union[str, gym.Env],
        eval_freq: int = 5_000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        best_model_path: Optional[Union[str, Path]] = None,
        reward_threshold: Optional[float] = None,
    ) -> None:
        self.eval_env = make_env(eval_env)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_path = best_model_path
        self.reward_threshold = reward_threshold
        self.best_mean_reward = -np.inf
        self.history: List[Tuple[int, float, float]] = []  # (timesteps, mean, std)
        self._last_eval = 0

    def on_training_start(self, agent: "BaseAgent") -> None:
        self._last_eval = agent.num_timesteps

    def on_step(self, agent: "BaseAgent") -> bool:
        if agent.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = agent.num_timesteps
        mean, std = evaluate_policy(
            agent, self.eval_env, self.n_eval_episodes, self.deterministic
        )
        self.history.append((agent.num_timesteps, mean, std))
        agent.logger.info(f"[eval] timesteps={agent.num_timesteps} mean={mean:.2f} std={std:.2f}")
        if mean > self.best_mean_reward:
            self.best_mean_reward = mean
            if self.best_model_path is not None:
                agent.save(self.best_model_path)
        if self.reward_threshold is not None and mean >= self.reward_threshold:
            agent.logger.info(f"[eval] reward threshold {self.reward_threshold} reached, stopping")
            return False
        return True


class CheckpointCallback(Callback):
    """Save the agent every ``save_freq`` timesteps to ``save_dir``."""

    def __init__(
        self, save_freq: int, save_dir: Union[str, Path], name_prefix: str = "relepy"
    ) -> None:
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.name_prefix = name_prefix
        self._last_save = 0

    def on_training_start(self, agent: "BaseAgent") -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_save = agent.num_timesteps

    def on_step(self, agent: "BaseAgent") -> bool:
        if agent.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = agent.num_timesteps
            agent.save(self.save_dir / f"{self.name_prefix}_{agent.num_timesteps}_steps.relepy")
        return True


class StopOnRewardThreshold(Callback):
    """Stop when the mean training return over recent episodes reaches ``reward_threshold``."""

    def __init__(self, reward_threshold: float, min_episodes: int = 20) -> None:
        self.reward_threshold = reward_threshold
        self.min_episodes = min_episodes

    def on_step(self, agent: "BaseAgent") -> bool:
        rewards: Any = agent.episode_rewards
        if len(rewards) >= self.min_episodes and float(np.mean(rewards)) >= self.reward_threshold:
            agent.logger.info(f"Reward threshold {self.reward_threshold} reached, stopping")
            return False
        return True
