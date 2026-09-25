"""SARSA: on-policy TD control."""

from __future__ import annotations

from relepy.algorithms.tabular._base import TabularAgent


class SARSA(TabularAgent):
    """SARSA. Target: ``r + gamma * Q(s', a')`` where ``a'`` is the action actually taken next."""

    def _td_target(
        self, reward: float, next_state: int, next_action: int, terminated: bool
    ) -> float:
        if terminated:
            return reward
        return reward + self.config.gamma * float(self.q_table[next_state, next_action])
