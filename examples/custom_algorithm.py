"""Writing a new algorithm on top of ReLePy's building blocks.

Double Q-learning (van Hasselt, 2010) in ~40 lines: the template ``StepwiseAgent`` already
handles the environment loop, vectorized collection, seeding, callbacks, logging and
evaluation; listing ``_state_attrs`` makes ``save`` / ``load`` work.
"""

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

import relepy
from relepy import GridWorldEnv, StepwiseAgent
from relepy.algorithms.tabular import TabularConfig


@dataclass
class DoubleQConfig(TabularConfig):
    pass  # same hyperparameters as Q-learning


@relepy.register("double_q_learning")
class DoubleQLearning(StepwiseAgent):
    config_class = DoubleQConfig
    _state_attrs = ("q_a", "q_b")

    def _setup(self) -> None:
        n_s, n_a = self.observation_space.n, self.action_space.n
        self.q_a = np.zeros((n_s, n_a))
        self.q_b = np.zeros((n_s, n_a))

    @property
    def epsilon(self) -> float:
        cfg = self.config
        frac = min(self.num_timesteps / max(cfg.epsilon_decay_steps, 1), 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def predict(self, observation: Any, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_space.n))
        q = self.q_a[int(observation)] + self.q_b[int(observation)]
        return int(np.argmax(q))

    def _select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.array([self.predict(o, deterministic=False) for o in obs])
        return actions, actions

    def _observe(self, tr: relepy.Transition) -> None:
        cfg = self.config
        for s, a, r, s2, term in zip(tr.obs, tr.actions, tr.rewards, tr.next_obs, tr.terminated):
            flip = self.rng.random() < 0.5
            update, other = (self.q_a, self.q_b) if flip else (self.q_b, self.q_a)
            best = int(np.argmax(update[s2]))
            target = r if term else r + cfg.gamma * other[s2, best]
            update[s, a] += cfg.alpha * (target - update[s, a])


if __name__ == "__main__":
    agent = relepy.make("double_q_learning", lambda: GridWorldEnv(size=5), n_envs=4,
                        alpha=0.3, epsilon_decay_steps=5_000, seed=0, verbose=0)
    agent.fit(20_000)
    print("mean return:", agent.evaluate(n_episodes=10))
    agent.save("double_q.relepy")
    print("reloaded:", type(relepy.load("double_q.relepy", GridWorldEnv(size=5))).__name__)
