"""Schedules map a timestep to a value (e.g. exploration rate, learning rate)."""

from __future__ import annotations

import math
from typing import Callable

Schedule = Callable[[int], float]


def constant_schedule(value: float) -> Schedule:
    """Always return ``value``."""
    return lambda step: value


def linear_schedule(start: float, end: float, duration: int) -> Schedule:
    """Linearly move from ``start`` to ``end`` over ``duration`` steps, then stay at ``end``."""
    if duration <= 0:
        return lambda step: end

    def schedule(step: int) -> float:
        frac = min(max(step / duration, 0.0), 1.0)
        return start + frac * (end - start)

    return schedule


def exponential_schedule(start: float, end: float, decay_steps: int) -> Schedule:
    """Exponentially decay from ``start`` towards ``end`` with time constant ``decay_steps``."""
    if decay_steps <= 0:
        return lambda step: end

    def schedule(step: int) -> float:
        return end + (start - end) * math.exp(-max(step, 0) / decay_steps)

    return schedule
