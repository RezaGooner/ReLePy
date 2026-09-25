"""Q-learning (Watkins, 1989): off-policy TD control."""

from __future__ import annotations

from relepy.algorithms.tabular._base import TabularAgent


class QLearning(TabularAgent):
    """Q-learning. Target: ``r + gamma * max_a' Q(s', a')``.

    Example:
        >>> agent = QLearning("FrozenLake-v1", alpha=0.2, seed=0)  # doctest: +SKIP
        >>> agent.fit(20_000).evaluate(n_episodes=20)  # doctest: +SKIP
    """

    def _td_target(
        self, reward: float, next_state: int, next_action: int, terminated: bool
    ) -> float:
        if terminated:
            return reward
        return reward + self.config.gamma * float(self.q_table[next_state].max())
