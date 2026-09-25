"""ReLePy: a modular reinforcement learning library.

Algorithms are imported lazily so that tabular methods work without PyTorch::

    from relepy import QLearning          # NumPy only
    from relepy import DQN, PPO           # requires ``pip install relepy[torch]``
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from relepy._version import __version__

_LAZY_ATTRS: Dict[str, str] = {
    # core (no torch)
    "BaseAgent": "relepy.core.base",
    "BaseConfig": "relepy.core.config",
    "DeepConfig": "relepy.core.config",
    "Callback": "relepy.core.callbacks",
    "CallbackList": "relepy.core.callbacks",
    "EvalCallback": "relepy.core.callbacks",
    "CheckpointCallback": "relepy.core.callbacks",
    "StopOnRewardThreshold": "relepy.core.callbacks",
    "evaluate_policy": "relepy.utils.evaluation",
    # tabular (no torch)
    "TabularConfig": "relepy.algorithms.tabular",
    "QLearning": "relepy.algorithms.tabular",
    "SARSA": "relepy.algorithms.tabular",
    "ExpectedSARSA": "relepy.algorithms.tabular",
    # value based (torch)
    "DQNConfig": "relepy.algorithms.value_based",
    "DQN": "relepy.algorithms.value_based",
    "DoubleDQN": "relepy.algorithms.value_based",
    "DuelingDQN": "relepy.algorithms.value_based",
    # policy gradient (torch)
    "REINFORCEConfig": "relepy.algorithms.policy_gradient",
    "REINFORCE": "relepy.algorithms.policy_gradient",
    "A2CConfig": "relepy.algorithms.policy_gradient",
    "A2C": "relepy.algorithms.policy_gradient",
    "PPOConfig": "relepy.algorithms.policy_gradient",
    "PPO": "relepy.algorithms.policy_gradient",
    # off-policy actor-critic (torch)
    "DDPGConfig": "relepy.algorithms.actor_critic",
    "DDPG": "relepy.algorithms.actor_critic",
    "TD3Config": "relepy.algorithms.actor_critic",
    "TD3": "relepy.algorithms.actor_critic",
    "SACConfig": "relepy.algorithms.actor_critic",
    "SAC": "relepy.algorithms.actor_critic",
    # sample environments
    "GridWorldEnv": "relepy.envs",
}

__all__ = ["__version__", *_LAZY_ATTRS]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'relepy' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
