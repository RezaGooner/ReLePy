"""Tabular temporal-difference methods (NumPy only)."""

from relepy.algorithms.tabular._base import TabularAgent, TabularConfig
from relepy.algorithms.tabular.expected_sarsa import ExpectedSARSA
from relepy.algorithms.tabular.q_learning import QLearning
from relepy.algorithms.tabular.sarsa import SARSA

__all__ = ["ExpectedSARSA", "QLearning", "SARSA", "TabularAgent", "TabularConfig"]
