"""PyTorch helpers. Importing this module requires PyTorch."""

from __future__ import annotations

from typing import Any, Union

import torch
from torch import nn


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA if available, otherwise CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """In-place soft update: ``target <- (1 - tau) * target + tau * source``."""
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def to_cpu(obj: Any) -> Any:
    """Recursively move tensors inside dicts/lists/tuples to CPU (used when saving)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return type(obj)((k, to_cpu(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj
