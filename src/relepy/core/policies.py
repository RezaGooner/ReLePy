"""Neural network modules. Requires PyTorch."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical, Distribution, Independent, Normal

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: Sequence[int],
    activation: str = "relu",
    orthogonal: bool = False,
    out_gain: float = 1.0,
) -> nn.Sequential:
    """Multi-layer perceptron ``in_dim -> hidden... -> out_dim`` (no activation on the output)."""
    act = _ACTIVATIONS[activation]
    layers: list = []
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    net = nn.Sequential(*layers)
    if orthogonal:
        linears = [m for m in net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)
        nn.init.zeros_(linears[-1].bias)
    return net


class QNetwork(nn.Module):
    """``obs -> Q(obs, .)``. With ``dueling=True`` uses separate value and advantage streams."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "relu",
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        if dueling:
            width = hidden_sizes[-1]
            self.trunk = nn.Sequential(
                build_mlp(obs_dim, width, hidden_sizes[:-1], activation), _ACTIVATIONS[activation]()
            )
            self.value_head = nn.Linear(width, 1)
            self.advantage_head = nn.Linear(width, n_actions)
        else:
            self.net = build_mlp(obs_dim, n_actions, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(obs)
        z = self.trunk(obs)
        adv = self.advantage_head(z)
        return self.value_head(z) + adv - adv.mean(dim=1, keepdim=True)


class ActorCriticPolicy(nn.Module):
    """Separate actor/critic MLPs for ``Discrete`` (categorical) or ``Box`` (Gaussian) actions."""

    def __init__(
        self,
        obs_dim: int,
        action_space: spaces.Space,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        log_std_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, spaces.Discrete)
        if self.discrete:
            out_dim = int(action_space.n)  # type: ignore[attr-defined]
        elif isinstance(action_space, spaces.Box):
            out_dim = int(np.prod(action_space.shape))
            self.log_std = nn.Parameter(torch.full((out_dim,), float(log_std_init)))
        else:
            raise TypeError(f"Unsupported action space {type(action_space).__name__}")
        self.actor = build_mlp(obs_dim, out_dim, hidden_sizes, activation, True, out_gain=0.01)
        self.critic = build_mlp(obs_dim, 1, hidden_sizes, activation, True, out_gain=1.0)

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        out = self.actor(obs)
        if self.discrete:
            return Categorical(logits=out)
        return Independent(Normal(out, self.log_std.exp().expand_as(out)), 1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def _mode(self, dist: Distribution) -> torch.Tensor:
        if self.discrete:
            return dist.probs.argmax(dim=-1)  # type: ignore[attr-defined]
        return dist.base_dist.mean  # type: ignore[attr-defined]

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(actions, values, log_probs)`` for a batch of observations."""
        dist = self.get_distribution(obs)
        actions = self._mode(dist) if deterministic else dist.sample()
        return actions, self.value(obs), dist.log_prob(actions)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(values, log_probs, entropy)`` of ``actions`` under the current policy."""
        dist = self.get_distribution(obs)
        return self.value(obs), dist.log_prob(actions), dist.entropy()


class DeterministicActor(nn.Module):
    """``obs -> action`` in ``[-1, 1]`` (tanh-squashed), used by DDPG and TD3."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, action_dim, hidden_sizes, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class SquashedGaussianActor(nn.Module):
    """Stochastic tanh-squashed Gaussian policy used by SAC. Actions lie in ``[-1, 1]``."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        width = hidden_sizes[-1]
        self.trunk = nn.Sequential(
            build_mlp(obs_dim, width, hidden_sizes[:-1], activation), _ACTIVATIONS[activation]()
        )
        self.mean_head = nn.Linear(width, action_dim)
        self.log_std_head = nn.Linear(width, action_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(action, log_prob)``; ``log_prob`` is ``None`` when deterministic."""
        z = self.trunk(obs)
        mean = self.mean_head(z)
        if deterministic:
            return torch.tanh(mean), None
        log_std = self.log_std_head(z).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        normal = Normal(mean, log_std.exp())
        u = normal.rsample()
        action = torch.tanh(u)
        # change of variables for the tanh squashing: log(1 - tanh(u)^2)
        correction = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        log_prob = normal.log_prob(u).sum(dim=-1) - correction.sum(dim=-1)
        return action, log_prob


class ContinuousCritic(nn.Module):
    """An ensemble of ``n_critics`` Q-networks ``(obs, action) -> Q``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
        n_critics: int = 2,
    ) -> None:
        super().__init__()
        self.q_nets = nn.ModuleList(
            build_mlp(obs_dim + action_dim, 1, hidden_sizes, activation) for _ in range(n_critics)
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([obs, actions], dim=1)
        return tuple(q(x) for q in self.q_nets)

    def q1(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q_nets[0](torch.cat([obs, actions], dim=1))

    def q_min(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Element-wise minimum over the ensemble, shape ``(B, 1)``."""
        return torch.cat(self(obs, actions), dim=1).min(dim=1, keepdim=True)[0]
