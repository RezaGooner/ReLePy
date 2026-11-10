"""Reproducible benchmark of ReLePy's algorithms.

Trains each (algorithm, environment) pair of a suite with several seeds, evaluates the final
policy and writes

* ``results.csv``  -- one row per run (final mean/std return, wall-clock time),
* ``curves.csv``   -- training curves (``rollout/ep_rew_mean`` per logged row),
* ``summary.md``   -- mean +/- std over seeds, ready to paste into a README or paper,
* ``curves.png``   -- learning curves (only if matplotlib is installed).

Usage::

    python benchmarks/run_benchmarks.py --suite quick              # ~minutes on a laptop CPU
    python benchmarks/run_benchmarks.py --suite full --seeds 0 1 2 3 4
    python benchmarks/run_benchmarks.py --only ppo sac --out results/

The suites are deliberately small classic-control tasks so that they run on CPU in CI.
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import numpy as np

import relepy


class Task(NamedTuple):
    algorithm: str
    env_id: str
    timesteps: int
    kwargs: Dict[str, Any]
    #: Mean return a correct implementation should exceed (used for pass/fail).
    threshold: float


FROZEN_LAKE = dict(alpha=0.1, epsilon_decay_steps=20_000)
SUITES: Dict[str, List[Task]] = {
    "quick": [
        Task("q_learning", "FrozenLake-v1", 40_000, FROZEN_LAKE, 0.5),
        Task("sarsa", "FrozenLake-v1", 40_000, FROZEN_LAKE, 0.5),
        Task("expected_sarsa", "FrozenLake-v1", 40_000, FROZEN_LAKE, 0.5),
        Task("double_dqn", "CartPole-v1", 40_000,
             dict(epsilon_decay_steps=8_000, target_update_interval=250), 100.0),
        Task("a2c", "CartPole-v1", 80_000, dict(n_envs=4, n_steps=8), 150.0),
        Task("ppo", "CartPole-v1", 60_000, dict(n_envs=4, n_steps=128, batch_size=64), 150.0),
        Task("sac", "Pendulum-v1", 15_000, dict(learning_starts=1_000), -700.0),
    ],
}
SUITES["full"] = SUITES["quick"] + [
    Task("dqn", "CartPole-v1", 50_000,
         dict(epsilon_decay_steps=10_000, target_update_interval=250), 100.0),
    Task("dueling_dqn", "CartPole-v1", 50_000,
         dict(double_dqn=True, epsilon_decay_steps=10_000, target_update_interval=250), 100.0),
    Task("reinforce", "CartPole-v1", 100_000, dict(episodes_per_update=4), 100.0),
    Task("ppo", "Pendulum-v1", 200_000,
         dict(n_envs=4, n_steps=512, batch_size=64, gae_lambda=0.95, gamma=0.9,
              learning_rate=1e-3), -700.0),
    Task("ddpg", "Pendulum-v1", 20_000, dict(learning_starts=1_000), -700.0),
    Task("td3", "Pendulum-v1", 20_000, dict(learning_starts=1_000), -700.0),
]


def run_task(task: Task, seed: int, n_eval_episodes: int) -> Dict[str, Any]:
    agent = relepy.make(task.algorithm, task.env_id, seed=seed, verbose=0, **task.kwargs)
    start = time.perf_counter()
    agent.fit(task.timesteps, log_interval=1)
    seconds = time.perf_counter() - start
    mean, std = agent.evaluate(n_episodes=n_eval_episodes, seed=10_000 + seed)
    curve = [
        (row["timesteps"], row["rollout/ep_rew_mean"])
        for row in agent.logger.history
        if "rollout/ep_rew_mean" in row
    ]
    return {"mean_return": mean, "std_return": std, "seconds": seconds, "curve": curve}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--only", nargs="*", help="restrict to these algorithm names")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    tasks = [t for t in SUITES[args.suite] if not args.only or t.algorithm in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    for task in tasks:
        for seed in args.seeds:
            res = run_task(task, seed, args.eval_episodes)
            label = f"{task.algorithm}/{task.env_id}"
            print(f"{label:32s} seed={seed}  return={res['mean_return']:9.2f}  "
                  f"time={res['seconds']:6.1f}s")
            rows.append({"algorithm": task.algorithm, "env": task.env_id, "seed": seed,
                         "timesteps": task.timesteps, "mean_return": res["mean_return"],
                         "std_return": res["std_return"], "seconds": res["seconds"]})
            curves += [{"algorithm": task.algorithm, "env": task.env_id, "seed": seed,
                        "timesteps": t, "ep_rew_mean": r} for t, r in res["curve"]]

    _write_csv(args.out / "results.csv", rows)
    _write_csv(args.out / "curves.csv", curves)
    summary = _summary(tasks, rows, args.seeds)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    _plot(args.out / "curves.png", tasks, curves)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(tasks: List[Task], rows: List[Dict[str, Any]], seeds: List[int]) -> str:
    lines = [
        f"ReLePy {relepy.__version__} | Python {platform.python_version()} | seeds {seeds}",
        "",
        "| Algorithm | Environment | Timesteps | Return (mean ± std over seeds) | Threshold "
        "| Time / run |",
        "|---|---|---|---|---|---|",
    ]
    for task in tasks:
        sel = [r for r in rows if r["algorithm"] == task.algorithm and r["env"] == task.env_id]
        means = np.array([r["mean_return"] for r in sel])
        ok = "pass" if means.mean() > task.threshold else "FAIL"
        lines.append(
            f"| {task.algorithm} | {task.env_id} | {task.timesteps:,} | "
            f"{means.mean():.2f} ± {means.std():.2f} | > {task.threshold:g} ({ok}) | "
            f"{np.mean([r['seconds'] for r in sel]):.1f}s |"
        )
    return "\n".join(lines) + "\n"


def _plot(path: Path, tasks: List[Task], curves: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping curves.png")
        return
    envs = sorted({t.env_id for t in tasks})
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 3.6), squeeze=False)
    for ax, env in zip(axes[0], envs):
        for algo in sorted({t.algorithm for t in tasks if t.env_id == env}):
            pts = [c for c in curves if c["env"] == env and c["algorithm"] == algo]
            if not pts:
                continue
            grid = np.linspace(0, max(c["timesteps"] for c in pts), 50)
            per_seed = []
            for seed in sorted({c["seed"] for c in pts}):
                s = [c for c in pts if c["seed"] == seed]
                per_seed.append(np.interp(grid, [c["timesteps"] for c in s],
                                          [c["ep_rew_mean"] for c in s]))
            mean, std = np.mean(per_seed, axis=0), np.std(per_seed, axis=0)
            ax.plot(grid, mean, label=algo)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        ax.set_title(env)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("training episode return")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
