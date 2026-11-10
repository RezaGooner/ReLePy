"""REINFORCE (Williams, 1992) with optional learned baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from relepy.core.losses import reinforce_loss
from relepy.core.on_policy import ActorCriticAgent, ActorCriticConfig

Episode = Dict[str, list]


@dataclass
class REINFORCEConfig(ActorCriticConfig):
    """REINFORCE hyperparameters.

    Attributes:
        episodes_per_update: Complete episodes gathered before each gradient step.
        use_baseline: Subtract a learned state-value baseline (variance reduction).
    """

    learning_rate: float = 1e-3
    episodes_per_update: int = 4
    use_baseline: bool = True
    entropy_coef: float = 0.01

    def validate(self) -> None:
        super().validate()
        if self.episodes_per_update < 1:
            raise ValueError("episodes_per_update must be >= 1")


def _new_episode() -> Episode:
    return {"obs": [], "act": [], "rew": []}


class REINFORCE(ActorCriticAgent):
    """Monte-Carlo policy gradient. Supports ``Discrete`` and ``Box`` action spaces.

    Updates happen on whole episodes, so ``fit(total_timesteps)`` may overshoot the requested
    number of steps by up to one batch of episodes. ``log_interval`` counts updates.
    Returns are pure Monte-Carlo: a time-limit truncation is treated like the episode end.
    """

    config_class = REINFORCEConfig
    config: REINFORCEConfig

    def _setup(self) -> None:
        super()._setup()
        self._running: List[Episode] = [_new_episode() for _ in range(self.n_envs)]

    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        cfg = self.config
        end = self.num_timesteps + total_timesteps
        n_updates = 0
        while self.num_timesteps < end:
            finished: List[Episode] = []
            while len(finished) < cfg.episodes_per_update:
                obs_vecs = self._preprocess_batch(self.collector.ensure_started())
                env_actions, raw, _, _ = self._act(obs_vecs)
                tr = self.collector.step(env_actions, raw)
                for i in range(self.n_envs):
                    ep = self._running[i]
                    ep["obs"].append(obs_vecs[i])
                    ep["act"].append(raw[i])
                    ep["rew"].append(float(tr.rewards[i]))
                    if tr.dones[i]:
                        finished.append(ep)
                        self._running[i] = _new_episode()
                if not self._on_step():
                    return
            stats = self._update(finished)
            n_updates += 1
            if n_updates % log_interval == 0:
                self._log_training(stats)

    def _discounted_returns(self, rewards: List[float]) -> List[float]:
        g, out = 0.0, []
        for r in reversed(rewards):
            g = r + self.config.gamma * g
            out.append(g)
        return out[::-1]

    def _update(self, episodes: List[Episode]) -> Dict[str, float]:
        cfg = self.config
        returns = [g for ep in episodes for g in self._discounted_returns(ep["rew"])]
        obs = torch.as_tensor(
            np.stack([o for ep in episodes for o in ep["obs"]]), dtype=torch.float32,
            device=self.device,
        )
        actions = self._action_tensor(np.stack([a for ep in episodes for a in ep["act"]]))
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        loss, stats = reinforce_loss(
            self.policy, obs, actions, returns_t, cfg.use_baseline, cfg.value_coef,
            cfg.entropy_coef, cfg.normalize_advantage,
        )
        self._optimize(loss)
        return stats
