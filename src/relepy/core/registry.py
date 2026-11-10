"""Algorithm registry: look agents up by name, build them, and load saved files generically.

    >>> import relepy
    >>> relepy.list_algorithms()  # doctest: +SKIP
    ['a2c', 'ddpg', 'double_dqn', ...]
    >>> agent = relepy.make("ppo", "CartPole-v1", n_steps=512)  # doctest: +SKIP
    >>> agent = relepy.load("model.relepy", "CartPole-v1")  # doctest: +SKIP

Built-in algorithms are registered by import path so that looking them up does not import
PyTorch until needed. Third-party agents register themselves with :func:`register`.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

if TYPE_CHECKING:  # pragma: no cover
    from relepy.core.base import BaseAgent

T = TypeVar("T")

# name -> "module:ClassName" (lazy) or the class itself (user registrations)
_REGISTRY: Dict[str, Union[str, type]] = {
    "q_learning": "relepy.algorithms.tabular:QLearning",
    "sarsa": "relepy.algorithms.tabular:SARSA",
    "expected_sarsa": "relepy.algorithms.tabular:ExpectedSARSA",
    "dqn": "relepy.algorithms.value_based:DQN",
    "double_dqn": "relepy.algorithms.value_based:DoubleDQN",
    "dueling_dqn": "relepy.algorithms.value_based:DuelingDQN",
    "reinforce": "relepy.algorithms.policy_gradient:REINFORCE",
    "a2c": "relepy.algorithms.policy_gradient:A2C",
    "ppo": "relepy.algorithms.policy_gradient:PPO",
    "ddpg": "relepy.algorithms.actor_critic:DDPG",
    "td3": "relepy.algorithms.actor_critic:TD3",
    "sac": "relepy.algorithms.actor_critic:SAC",
}
_ALIASES: Dict[str, str] = {"qlearning": "q_learning", "expectedsarsa": "expected_sarsa",
                            "doubledqn": "double_dqn", "duelingdqn": "dueling_dqn"}


def _normalize(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key.replace("_", ""), _ALIASES.get(key, key))


def register(name: str, cls: Any = None, *, overwrite: bool = False) -> Any:
    """Register an agent class under ``name``. Usable as a decorator::

        @relepy.register("my_algo")
        class MyAlgo(relepy.BaseAgent): ...
    """

    def decorator(klass: Type[T]) -> Type[T]:
        key = _normalize(name)
        if key in _REGISTRY and not overwrite and _REGISTRY[key] is not klass:
            raise ValueError(f"Algorithm {key!r} is already registered (use overwrite=True)")
        _REGISTRY[key] = klass
        return klass

    return decorator(cls) if cls is not None else decorator


def get_algorithm(name: str) -> Type["BaseAgent"]:
    """Return the agent class registered under ``name`` (also accepts class names)."""
    key = _normalize(name)
    if key not in _REGISTRY:
        for candidate in list(_REGISTRY):  # allow class names such as "DoubleDQN"
            target = _REGISTRY[candidate]
            cls_name = target.rsplit(":", 1)[1] if isinstance(target, str) else target.__name__
            if cls_name.lower() == name.lower():
                key = candidate
                break
        else:
            raise KeyError(f"Unknown algorithm {name!r}. Available: {list_algorithms()}")
    target = _REGISTRY[key]
    if isinstance(target, str):
        module, cls_name = target.split(":")
        target = getattr(importlib.import_module(module), cls_name)
        _REGISTRY[key] = target
    return target


def list_algorithms() -> List[str]:
    """Names of all registered algorithms."""
    return sorted(_REGISTRY)


def make(name: str, env: Any, *args: Any, **kwargs: Any) -> "BaseAgent":
    """Build an agent by name: ``make("ppo", "CartPole-v1", seed=0, n_steps=512)``."""
    return get_algorithm(name)(env, *args, **kwargs)


def load(path: Union[str, Path], env: Any, **kwargs: Any) -> "BaseAgent":
    """Load any saved agent; the algorithm is read from the file."""
    from relepy.core.serialization import read_metadata

    algorithm = read_metadata(path)["algorithm"]
    cls = get_algorithm(algorithm)
    return cls.load(path, env, **kwargs)

