"""Benchmark ReLePy against Stable-Baselines3 on the same task and budget.

Runs a few seeds of ReLePy's PPO and (if installed) SB3's PPO on CartPole-v1 with matched
hyperparameters and timesteps, then prints mean +/- std final evaluation return and wall-clock
training time for each. Useful for the reproducibility section of a paper.

    pip install "relepy[torch]" stable-baselines3
    python examples/benchmark_vs_sb3.py
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

ENV_ID = "CartPole-v1"
TOTAL_TIMESTEPS = 50_000
SEEDS = [0, 1, 2]
COMMON_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


def run_relepy() -> List[Dict[str, float]]:
    from relepy import PPO

    results = []
    for seed in SEEDS:
        agent = PPO(
            ENV_ID,
            learning_rate=COMMON_KWARGS["learning_rate"],
            n_steps=COMMON_KWARGS["n_steps"],
            batch_size=COMMON_KWARGS["batch_size"],
            n_epochs=COMMON_KWARGS["n_epochs"],
            gamma=COMMON_KWARGS["gamma"],
            gae_lambda=COMMON_KWARGS["gae_lambda"],
            clip_range=COMMON_KWARGS["clip_range"],
            entropy_coef=COMMON_KWARGS["ent_coef"],
            seed=seed,
            verbose=0,
        )
        start = time.perf_counter()
        agent.fit(TOTAL_TIMESTEPS)
        elapsed = time.perf_counter() - start
        mean, std = agent.evaluate(n_episodes=20, seed=seed)
        results.append({"seed": seed, "mean_return": mean, "std_return": std, "seconds": elapsed})
    return results


def run_sb3() -> List[Dict[str, float]]:
    import gymnasium as gym
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    results = []
    for seed in SEEDS:
        env = gym.make(ENV_ID)
        model = SB3PPO(
            "MlpPolicy",
            env,
            learning_rate=COMMON_KWARGS["learning_rate"],
            n_steps=COMMON_KWARGS["n_steps"],
            batch_size=COMMON_KWARGS["batch_size"],
            n_epochs=COMMON_KWARGS["n_epochs"],
            gamma=COMMON_KWARGS["gamma"],
            gae_lambda=COMMON_KWARGS["gae_lambda"],
            clip_range=COMMON_KWARGS["clip_range"],
            ent_coef=COMMON_KWARGS["ent_coef"],
            seed=seed,
            verbose=0,
        )
        start = time.perf_counter()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        elapsed = time.perf_counter() - start
        mean, std = evaluate_policy(model, gym.make(ENV_ID), n_eval_episodes=20)
        results.append({"seed": seed, "mean_return": mean, "std_return": std, "seconds": elapsed})
    return results


def summarize(name: str, results: List[Dict[str, float]]) -> None:
    means = [r["mean_return"] for r in results]
    secs = [r["seconds"] for r in results]
    print(f"{name}: return {np.mean(means):.1f} +/- {np.std(means):.1f} "
          f"(seeds={SEEDS}), time {np.mean(secs):.1f}s/run")


if __name__ == "__main__":
    print(f"Benchmarking PPO on {ENV_ID} for {TOTAL_TIMESTEPS} timesteps, seeds={SEEDS}\n")
    summarize("ReLePy", run_relepy())
    try:
        summarize("Stable-Baselines3", run_sb3())
    except ImportError:
        print("Stable-Baselines3 not installed; skipping (pip install stable-baselines3)")
