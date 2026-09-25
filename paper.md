---
title: 'ReLePy: A Modular Reinforcement Learning Library with a Consistent, Configurable API'
tags:
  - Python
  - reinforcement learning
  - machine learning
  - deep learning
  - PyTorch
authors:
  - name: Reza Asadi
    orcid: 0009-0005-6852-5756
    affiliation: 1
affiliations:
  - name: "Independent Researcher"
    index: 1
date: 25 September 2026
bibliography: paper.bib
---

# Summary

`ReLePy` is a Python library that collects a broad range of reinforcement learning (RL)
algorithms behind a single, consistent interface: every agent is built, trained, evaluated,
saved and loaded through the same four methods (`fit`, `predict`, `evaluate`, `save`/`load`),
regardless of whether it is a classical tabular method or a deep neural network trained with
PyTorch. The library currently implements three tabular temporal-difference algorithms
(Q-learning, SARSA, Expected SARSA), three value-based deep methods (DQN and its Double and
Dueling variants), three on-policy policy-gradient methods (REINFORCE, A2C, PPO), and three
off-policy actor-critic methods for continuous control (DDPG, TD3, SAC). Every algorithm has a
dedicated, type-annotated, validated configuration object (a Python `dataclass`) that exposes
its hyperparameters explicitly and can be serialized to and from JSON, which makes experiments
easy to reproduce and to compare. `ReLePy` interoperates with any [Gymnasium](https://gymnasium.farama.org/)
environment [@towers2024gymnasium], supports vectorized (parallel) training for its on-policy
algorithms, and stores trained models in a pickle-free archive format so that loading a saved
agent never executes arbitrary code.

# Statement of need

Reinforcement learning research and teaching both benefit from software that lets a user move
between very different families of algorithms -- tabular, value-based and policy-based -- without
learning a new interface each time. Existing mature libraries such as
Stable-Baselines3 [@raffin2021stable] and CleanRL [@huang2022cleanrl] are excellent resources,
but they embody two different philosophies: Stable-Baselines3 favours a unified, production-style
API but only implements deep, PyTorch-based algorithms, so a user who also wants classical
tabular Q-learning or SARSA has to switch to a separate package; CleanRL favours single-file,
easily readable reference implementations, at the cost of code duplication across algorithms and
the absence of a shared, reusable base class. `ReLePy` is designed to sit between these two
philosophies: it keeps one shared agent interface and one configuration pattern across *all*
algorithm families, including tabular methods that require no deep learning framework at all,
while still keeping each algorithm's implementation in its own short, readable module. This
makes `ReLePy` suitable for teaching settings where students progress from tabular control to
deep value-based and policy-gradient methods within one consistent codebase, and for practitioners
who want to prototype quickly with an explicit, reproducible hyperparameter configuration.

`ReLePy` targets researchers, students and practitioners who need to (i) compare algorithms
across the tabular/deep-value/policy-gradient/actor-critic spectrum within one API, (ii) keep
hyperparameter configurations explicit, validated and serializable for reproducible experiments,
and (iii) load saved models without exposing themselves to the arbitrary code execution risk that
`pickle`-based formats carry, which is a documented pitfall of several existing RL and ML model
formats.

# Comparison to related work

The table below summarizes how `ReLePy` relates to two widely used RL libraries along the axes
most relevant to the design goals above.

| | ReLePy | Stable-Baselines3 | CleanRL |
|---|---|---|---|
| Tabular algorithms (Q-learning, SARSA) | Yes | No | No |
| Deep value-based / policy-gradient / actor-critic | Yes | Yes | Yes |
| Shared base class across all algorithms | Yes | Yes (deep only) | No (single-file design) |
| Typed, validated, JSON-serializable configs | Yes | Partial (dict-based) | No (argparse flags) |
| Vectorized (parallel) environment training | Yes (on-policy) | Yes | Yes |
| Pickle-free saved model format | Yes | No | N/A |

`ReLePy`'s PPO implementation was benchmarked against Stable-Baselines3's PPO on the classic
control task CartPole-v1 with matched hyperparameters (learning rate, rollout length, batch
size, number of epochs, discount factor, GAE lambda, clip range) and an identical training
budget of 50,000 environment steps, averaged over three random seeds. Both implementations
reached the maximum possible return of 500 with zero variance across seeds, while `ReLePy`
completed training in approximately 44% of the wall-clock time of Stable-Baselines3 on the same
machine, indicating that the added flexibility of `ReLePy`'s design does not come at the cost of
either final policy quality or training speed on this benchmark. The benchmark script is
included in the repository (`examples/benchmark_vs_sb3.py`) so that it can be reproduced or
extended to further tasks.

# Design and implementation

Every `ReLePy` agent subclasses a common `BaseAgent` abstract base class that implements the
public `fit`/`predict`/`evaluate`/`save`/`load` interface and delegates the algorithm-specific
learning update to subclasses. Hyperparameters are represented as `dataclass`-based configuration
objects with a `validate` method that raises immediately on out-of-range or inconsistent values,
and which support serialization to and from JSON for recording exact experimental settings.
Training progress can be monitored and controlled with a small callback system (evaluation on a
held-out environment, checkpointing, and early stopping on a reward threshold), modeled after
similar mechanisms in other RL libraries. Saved models are written as a ZIP archive containing
JSON metadata and NumPy `.npy` arrays rather than a `pickle` file, so that loading a model
downloaded from an untrusted source cannot execute arbitrary code -- a property not shared by
several existing RL and general machine-learning serialization formats.

# Acknowledgements

We acknowledge the authors of Gymnasium, PyTorch and Stable-Baselines3, whose design choices and
documentation informed several aspects of this library.

# References
