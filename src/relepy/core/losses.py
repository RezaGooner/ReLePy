"""Loss functions of every algorithm, as plain functions (requires PyTorch).

Keeping the maths out of the agent classes makes each objective easy to read, test and reuse:
a research variant usually only needs a new loss, not a new training loop.
Every function returns the scalar loss tensor (plus, where useful, a dict of statistics).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from relepy.core.buffers import ReplayBatch, RolloutBatch
from relepy.core.policies import (
    ActorCriticPolicy,
    ContinuousCritic,
    DeterministicActor,
    SquashedGaussianActor,
)

Stats = Dict[str, float]


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize to zero mean / unit std (no-op for a single element)."""
    if x.numel() <= 1:
        return x
    return (x - x.mean()) / (x.std() + eps)


def td_target(
    rewards: torch.Tensor, dones: torch.Tensor, next_values: torch.Tensor, gamma: float
) -> torch.Tensor:
    """One-step bootstrapped target ``r + gamma * (1 - done) * V(s')``."""
    return rewards + gamma * (1.0 - dones) * next_values


# ---------------------------------------------------------------------- value-based
def dqn_loss(
    q_net: nn.Module,
    target_net: nn.Module,
    batch: ReplayBatch,
    gamma: float,
    double_dqn: bool = False,
    loss: str = "huber",
) -> torch.Tensor:
    """DQN loss; with ``double_dqn`` the online net selects and the target net evaluates."""
    with torch.no_grad():
        if double_dqn:
            best = q_net(batch.next_observations).argmax(dim=1, keepdim=True)
            next_q = target_net(batch.next_observations).gather(1, best)
        else:
            next_q = target_net(batch.next_observations).max(dim=1, keepdim=True)[0]
        target = td_target(batch.rewards, batch.dones, next_q, gamma)
    current = q_net(batch.observations).gather(1, batch.actions)
    loss_fn = F.smooth_l1_loss if loss == "huber" else F.mse_loss
    return loss_fn(current, target)


# ------------------------------------------------------------------ policy gradient
def reinforce_loss(
    policy: ActorCriticPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    use_baseline: bool = True,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    normalize_advantage: bool = True,
) -> Tuple[torch.Tensor, Stats]:
    """Monte-Carlo policy gradient with an optional learned state-value baseline."""
    values, log_prob, entropy = policy.evaluate_actions(observations, actions)
    if use_baseline:
        adv = returns - values.detach()
        value_loss = F.mse_loss(values, returns)
    else:
        adv = returns
        value_loss = torch.zeros((), device=returns.device)
    if normalize_advantage:
        adv = normalize(adv)
    policy_loss = -(adv * log_prob).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    return loss, {
        "train/policy_loss": policy_loss.item(),
        "train/value_loss": value_loss.item(),
        "train/entropy": entropy.mean().item(),
    }


def a2c_loss(
    policy: ActorCriticPolicy,
    batch: RolloutBatch,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    normalize_advantage: bool = True,
) -> Tuple[torch.Tensor, Stats]:
    """Advantage actor-critic loss."""
    values, log_prob, entropy = policy.evaluate_actions(batch.observations, batch.actions)
    adv = normalize(batch.advantages) if normalize_advantage else batch.advantages
    policy_loss = -(adv * log_prob).mean()
    value_loss = F.mse_loss(values, batch.returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    return loss, {
        "train/policy_loss": policy_loss.item(),
        "train/value_loss": value_loss.item(),
        "train/entropy": entropy.mean().item(),
    }


def ppo_loss(
    policy: ActorCriticPolicy,
    batch: RolloutBatch,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    normalize_advantage: bool = True,
    clip_range_vf: Optional[float] = None,
) -> Tuple[torch.Tensor, Stats]:
    """PPO clipped surrogate loss. Stats include ``approx_kl`` (``KL(old || new)`` estimate)."""
    values, log_prob, entropy = policy.evaluate_actions(batch.observations, batch.actions)
    adv = normalize(batch.advantages) if normalize_advantage else batch.advantages
    log_ratio = log_prob - batch.old_log_probs
    ratio = torch.exp(log_ratio)
    surr1 = adv * ratio
    surr2 = adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = -torch.min(surr1, surr2).mean()
    if clip_range_vf is None:
        value_loss = F.mse_loss(values, batch.returns)
    else:
        delta = (values - batch.old_values).clamp(-clip_range_vf, clip_range_vf)
        clipped = batch.old_values + delta
        value_loss = torch.max(
            (values - batch.returns) ** 2, (clipped - batch.returns) ** 2
        ).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean().item()
    return loss, {
        "train/policy_loss": policy_loss.item(),
        "train/value_loss": value_loss.item(),
        "train/entropy": entropy.mean().item(),
        "train/approx_kl": approx_kl,
        "train/clip_fraction": clip_fraction,
    }


# ------------------------------------------------------- off-policy actor-critic
def _ensemble_mse(q_values: Tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
    """Sum of the MSE of every critic of an ensemble against the same target."""
    return torch.stack([F.mse_loss(q, target) for q in q_values]).sum()


def deterministic_critic_loss(
    critic: ContinuousCritic,
    critic_target: ContinuousCritic,
    actor_target: DeterministicActor,
    batch: ReplayBatch,
    gamma: float,
    target_policy_noise: float = 0.0,
    target_noise_clip: float = 0.5,
) -> torch.Tensor:
    """Critic loss of DDPG / TD3 (target smoothing noise and min over critics for TD3)."""
    with torch.no_grad():
        next_actions = actor_target(batch.next_observations)
        if target_policy_noise > 0:
            noise = (torch.randn_like(next_actions) * target_policy_noise).clamp(
                -target_noise_clip, target_noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
        next_q = critic_target.q_min(batch.next_observations, next_actions)
        target = td_target(batch.rewards, batch.dones, next_q, gamma)
    return _ensemble_mse(critic(batch.observations, batch.actions), target)


def deterministic_actor_loss(
    critic: ContinuousCritic, actor: DeterministicActor, observations: torch.Tensor
) -> torch.Tensor:
    """DDPG / TD3 actor loss ``-Q1(s, mu(s))``."""
    return -critic.q1(observations, actor(observations)).mean()


def sac_critic_loss(
    critic: ContinuousCritic,
    critic_target: ContinuousCritic,
    actor: SquashedGaussianActor,
    batch: ReplayBatch,
    gamma: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Soft Bellman residual of the twin critics."""
    with torch.no_grad():
        next_actions, next_log_prob = actor(batch.next_observations)
        next_q = critic_target.q_min(batch.next_observations, next_actions)
        next_q = next_q - alpha * next_log_prob.unsqueeze(1)
        target = td_target(batch.rewards, batch.dones, next_q, gamma)
    return _ensemble_mse(critic(batch.observations, batch.actions), target)


def sac_actor_loss(
    critic: ContinuousCritic,
    actor: SquashedGaussianActor,
    observations: torch.Tensor,
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(actor_loss, log_prob)``; ``log_prob`` feeds the temperature loss."""
    actions, log_prob = actor(observations)
    q_new = critic.q_min(observations, actions)
    return (alpha * log_prob.unsqueeze(1) - q_new).mean(), log_prob


def sac_temperature_loss(
    log_alpha: torch.Tensor, log_prob: torch.Tensor, target_entropy: float
) -> torch.Tensor:
    """Automatic entropy tuning: drive the policy entropy towards ``target_entropy``."""
    return -(log_alpha * (log_prob.detach() + target_entropy)).mean()
