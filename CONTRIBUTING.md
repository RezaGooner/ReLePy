# Contributing to ReLePy

Thanks for your interest! Bug reports, documentation fixes and new algorithms are all welcome.

## Development setup

```bash
git clone https://github.com/RezaGooner/ReLePy.git
cd ReLePy
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
```

## Checks (the same as CI)

```bash
ruff check src tests examples benchmarks
mypy src
pytest                 # fast tests
pytest -m slow         # learning tests (several minutes)
```

## Adding an algorithm

1. Pick the template closest to your algorithm (see [docs/index.md](docs/index.md)):
   `StepwiseAgent` (act and learn every step), `OffPolicyAgent` (replay buffer),
   `OnPolicyAgent` (rollouts + GAE) or `BaseAgent` (anything else).
2. Write a `*Config` dataclass deriving from the matching config and extend `validate()`.
3. Put the objective in `relepy/core/losses.py` as a plain function (easy to test and reuse).
4. List the learned attributes in `_state_attrs` so `save` / `load` work.
5. Register it in `relepy/core/registry.py` and `relepy/__init__.py` (`_LAZY_ATTRS`).
6. Add tests:
   * a fast smoke test (train a few hundred steps, `predict`, `save` / `load`, `n_envs > 1`);
   * a `@pytest.mark.slow` learning test showing it solves a standard task;
   * an entry in `benchmarks/run_benchmarks.py`.
7. Document it in the README table and add a line to `CHANGELOG.md` under *Unreleased*.

Every algorithm should cite its paper in the module docstring.

## Pull requests

* One topic per pull request; keep the public API backward compatible when possible and list
  breaking changes in the changelog.
* Code style: ruff, type hints on public functions, docstrings on public classes.

## Releasing (maintainers)

1. Update `src/relepy/_version.py` and move the *Unreleased* changelog entries under the new
   version.
2. Commit, then tag and push: `git tag v0.X.Y && git push origin v0.X.Y`.
3. The *Release* workflow tests, builds and publishes to PyPI (Trusted Publishing) and creates
   the GitHub release.
