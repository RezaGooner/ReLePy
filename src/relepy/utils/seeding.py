"""Global seeding utilities."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int]) -> np.random.Generator:
    """Seed ``random``, NumPy and (if installed) PyTorch, and return a NumPy ``Generator``.

    Passing ``None`` leaves the global RNGs untouched and returns an unseeded generator.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:  # PyTorch is optional
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover
            pass
    return np.random.default_rng(seed)
