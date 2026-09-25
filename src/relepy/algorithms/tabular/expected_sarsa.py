"""Expected SARSA: lower-variance on-policy TD control."""

from __future__ import annotations

from relepy.algorithms.tabular._base import TabularAgent


class ExpectedSARSA(TabularAgent):
    """Expected SARSA. Target: expectation of ``Q(s', .)`` under the epsilon-greedy policy."""

    def _td_target(
        self, reward: float, next_state: int, next_action: int, terminated: bool
    ) -> float:
        if terminated:
            return reward
        q = self.q_table[next_state]
        eps = self.epsilon
        expected = eps * float(q.mean()) + (1.0 - eps) * float(q.max())
        return reward + self.config.gamma * expected
