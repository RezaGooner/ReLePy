"""REINFORCE (Williams, 1992) with optional learned baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from relepy.algorithms.policy_gradient._base import ActorCriticAgent, ActorCriticConfig


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


class REINFORCE(ActorCriticAgent):
    """Monte-Carlo policy gradient. Supports ``Discrete`` and ``Box`` action spaces.

    Updates happen on whole episodes, so ``fit(total_timesteps)`` may overshoot the requested
    number of steps by up to one episode batch. ``log_interval`` counts updates.
    """

    config_class = REINFORCEConfig
    config: REINFORCEConfig

    def _learn(self, total_timesteps: int, log_interval: int) -> None:
        cfg = self.config
        end = self.num_timesteps + total_timesteps
        n_updates = 0
        while self.num_timesteps < end:
            episodes: List[Dict[str, list]] = []
            while len(episodes) < cfg.episodes_per_update:
                ep: Dict[str, list] = {"obs": [], "act": [], "rew": []}
                done = False
                while not done:
                    obs_vec = self._preprocess(self._obs)
                    env_action, raw, _, _ = self._act(obs_vec)
                    next_obs, reward, terminated, truncated, _ = self._env_step(env_action)
                    ep["obs"].append(obs_vec)
                    ep["act"].append(raw)
                    ep["rew"].append(reward)
                    done = terminated or truncated
                    if done:
                        self._reset_env()
                    else:
                        self._obs = next_obs
                    if not self._on_step():
                        return
                episodes.append(ep)
            stats = self._update(episodes)
            n_updates += 1
            if n_updates % log_interval == 0:
                self._log_training(stats)

    def _update(self, episodes: List[Dict[str, list]]) -> Dict[str, float]:
        cfg = self.config
        returns: List[float] = []
        for ep in episodes:
            g = 0.0
            ep_returns = []
            for r in reversed(ep["rew"]):
                g = r + cfg.gamma * g
                ep_returns.append(g)
            returns.extend(reversed(ep_returns))

        obs = torch.as_tensor(
            np.stack([o for ep in episodes for o in ep["obs"]]), dtype=torch.float32,
            device=self.device,
        )
        actions = self._action_tensor(np.stack([a for ep in episodes for a in ep["act"]]))
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        if cfg.use_baseline:
            adv = returns_t - values.detach()
            value_loss = F.mse_loss(values, returns_t)
        else:
            adv = returns_t
            value_loss = torch.zeros((), device=self.device)
        if cfg.normalize_advantage and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(adv * log_prob).mean()
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
        self._optimize(loss)
        return {
            "train/policy_loss": policy_loss.item(),
            "train/value_loss": value_loss.item(),
            "train/entropy": entropy.mean().item(),
        }
