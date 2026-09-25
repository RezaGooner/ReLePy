"""Hyperparameter configuration objects.

Every algorithm has a dataclass config with type hints, defaults and validation. Configs can be
created, updated, serialized to JSON and loaded again, which makes experiments reproducible::

    cfg = DQNConfig(learning_rate=5e-4, double_dqn=True)
    cfg2 = cfg.update(batch_size=128)
    cfg.to_json("cfg.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

T = TypeVar("T", bound="BaseConfig")

_ACTIVATIONS = ("relu", "tanh", "elu")


@dataclass
class BaseConfig:
    """Parameters common to every algorithm.

    Attributes:
        gamma: Discount factor in ``[0, 1]``.
    """

    gamma: float = 0.99

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for invalid values. Subclasses extend this and call ``super()``."""
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown hyperparameter(s) for {cls.__name__}: {sorted(unknown)}. "
                f"Valid names: {sorted(valid)}"
            )
        return cls(**data)

    def update(self: T, **changes: Any) -> T:
        """Return a copy with ``changes`` applied (validated)."""
        valid = {f.name for f in fields(self)}
        unknown = set(changes) - valid
        if unknown:
            raise ValueError(
                f"Unknown hyperparameter(s) for {type(self).__name__}: {sorted(unknown)}. "
                f"Valid names: {sorted(valid)}"
            )
        return replace(self, **changes)

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls: Type[T], source: Union[str, Path]) -> T:
        """Load from a JSON file path or a JSON string."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


@dataclass
class DeepConfig(BaseConfig):
    """Parameters shared by neural-network based algorithms.

    Attributes:
        learning_rate: Adam learning rate.
        hidden_sizes: Width of each hidden layer of the MLPs.
        activation: ``"relu"``, ``"tanh"`` or ``"elu"``.
        max_grad_norm: Gradient clipping threshold (``None`` disables clipping).
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
    """

    learning_rate: float = 3e-4
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "relu"
    max_grad_norm: Optional[float] = None
    device: str = "auto"

    def validate(self) -> None:
        super().validate()
        self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)  # JSON gives lists
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive integers, got {self.hidden_sizes}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0 or None, got {self.max_grad_norm}")
