"""A minimal logger that prints training statistics and keeps them in memory."""

from __future__ import annotations

from typing import Any, Dict, List


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class Logger:
    """Collects ``record``-ed scalars and flushes them with ``dump``.

    Every dumped row is kept in :attr:`history` (a list of dicts), which makes it easy to plot
    learning curves or export results afterwards, e.g. ``pandas.DataFrame(agent.logger.history)``.

    Args:
        verbose: ``0`` silent, ``1`` print training rows and info messages.
    """

    def __init__(self, verbose: int = 1) -> None:
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []
        self._current: Dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self._current[key] = value

    def dump(self, step: int) -> None:
        if not self._current:
            return
        row: Dict[str, Any] = {"timesteps": step, **self._current}
        self.history.append(row)
        if self.verbose >= 1:
            print(" | ".join(f"{k}={_fmt(v)}" for k, v in row.items()))
        self._current = {}

    def info(self, message: str) -> None:
        if self.verbose >= 1:
            print(message)
