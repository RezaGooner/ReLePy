"""Tests of the 0.2 building blocks: vector env, collector, registry, serialization."""

import subprocess
import sys
import zipfile

import gymnasium as gym
import numpy as np
import pytest

import relepy
from relepy import GridWorldEnv, QLearning, SyncVecEnv, make_vec_env
from relepy.core.collector import Collector
from relepy.core.serialization import LegacyFormatError, load_archive, save_archive


# ------------------------------------------------------------------------ vector env
def test_vec_env_autoreset_returns_final_obs():
    vec = SyncVecEnv([GridWorldEnv(size=2, max_steps=50) for _ in range(2)])
    vec.reset(seed=0)
    out = vec.step([1, 3])  # env 0: right -> (0,1); env 1: left -> wall, stays at 0
    np.testing.assert_array_equal(out.obs, [1, 0])
    out = vec.step([2, 1])  # env 0 reaches the goal (index 3) and is reset to 0
    assert out.terminated.tolist() == [True, False]
    assert out.final_obs[0] == 3 and out.obs[0] == 0
    assert out.obs[1] == out.final_obs[1] == 1


def test_make_vec_env_inputs():
    assert make_vec_env("CartPole-v1", 3).n_envs == 3
    assert make_vec_env(lambda: gym.make("CartPole-v1"), 2).n_envs == 2
    assert make_vec_env(GridWorldEnv(), 1).n_envs == 1
    with pytest.raises(ValueError, match="factory"):
        make_vec_env(GridWorldEnv(), 2)
    with pytest.raises(TypeError):
        make_vec_env(42, 1)


def test_collector_counts_steps_and_episodes():
    vec = SyncVecEnv([GridWorldEnv(size=2, max_steps=3) for _ in range(4)])
    col = Collector(vec, seed=0)
    for _ in range(3):  # always "up" -> never reaches the goal -> truncated after 3 steps
        tr = col.step(np.zeros(4, dtype=np.int64))
    assert col.num_timesteps == 12
    assert col.n_episodes == 4
    assert tr.truncated.all() and not tr.terminated.any()
    assert list(col.episode_lengths) == [3, 3, 3, 3]
    np.testing.assert_allclose(list(col.episode_rewards), [-0.03] * 4)


def test_vectorized_tabular_agent_learns():
    agent = QLearning(lambda: GridWorldEnv(size=4), n_envs=4, alpha=0.3,
                      epsilon_decay_steps=3_000, seed=0, verbose=0)
    agent.fit(12_000)
    assert agent.num_timesteps >= 12_000
    mean, _ = agent.evaluate(n_episodes=5)
    assert mean > 0.8


# -------------------------------------------------------------------------- registry
def test_registry_make_get_and_names():
    assert "ppo" in relepy.list_algorithms() and "q_learning" in relepy.list_algorithms()
    assert relepy.get_algorithm("QLearning") is QLearning
    assert relepy.get_algorithm("q-learning") is QLearning
    agent = relepy.make("q_learning", GridWorldEnv(), verbose=0)
    assert isinstance(agent, QLearning)
    with pytest.raises(KeyError, match="Unknown algorithm"):
        relepy.get_algorithm("does_not_exist")


def test_register_custom_agent():
    @relepy.register("my_qlearning_variant")
    class MyQ(QLearning):
        pass

    assert relepy.get_algorithm("my_qlearning_variant") is MyQ
    with pytest.raises(ValueError, match="already registered"):
        relepy.register("my_qlearning_variant", QLearning)


def test_generic_load(tmp_path):
    agent = QLearning(GridWorldEnv(), seed=0, verbose=0).fit(500)
    path = tmp_path / "q.relepy"
    agent.save(path)
    loaded = relepy.load(path, GridWorldEnv(), verbose=0)
    assert type(loaded) is QLearning
    np.testing.assert_allclose(loaded.q_table, agent.q_table)


# --------------------------------------------------------------------- serialization
def test_archive_roundtrip_nested_structures(tmp_path):
    state = {
        "table": np.arange(6.0).reshape(2, 3),
        "nested": {0: {"step": 3, "betas": (0.9, 0.999)}, "params": [0, 1, 2], "none": None},
        "flag": True,
    }
    path = tmp_path / "x.relepy"
    save_archive(path, {"algorithm": "Test"}, state)
    meta, loaded = load_archive(path)
    assert meta["algorithm"] == "Test" and meta["format_version"] == 1
    np.testing.assert_array_equal(loaded["table"], state["table"])
    assert loaded["nested"][0] == {"step": 3, "betas": (0.9, 0.999)}  # int key + tuple kept
    assert loaded["nested"]["params"] == [0, 1, 2] and loaded["nested"]["none"] is None
    assert loaded["flag"] is True


def test_saved_file_contains_no_pickle(tmp_path):
    path = tmp_path / "q.relepy"
    QLearning(GridWorldEnv(), verbose=0).fit(100).save(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert "metadata.json" in names and "state.json" in names
        for name in names:
            if name.endswith(".npy"):
                np.load(zf.open(name), allow_pickle=False)  # raises if pickled objects


def test_legacy_pickle_files_need_opt_in(tmp_path):
    import pickle

    agent = QLearning(GridWorldEnv(), seed=0, verbose=0).fit(300)
    legacy = {
        "relepy_version": "0.1.0",
        "algorithm": "QLearning",
        "config": {k: v for k, v in agent.config.to_dict().items() if k != "n_envs"},
        "num_timesteps": agent.num_timesteps,
        "state": {"q_table": agent.q_table.copy()},
    }
    path = tmp_path / "old.relepy"
    path.write_bytes(pickle.dumps(legacy))
    with pytest.raises(LegacyFormatError):
        QLearning.load(path, GridWorldEnv(), verbose=0)
    loaded = QLearning.load(path, GridWorldEnv(), verbose=0, allow_legacy_pickle=True)
    np.testing.assert_allclose(loaded.q_table, agent.q_table)


def test_tabular_works_without_torch():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from relepy import QLearning, GridWorldEnv\n"
        "a = QLearning(GridWorldEnv(), verbose=0, seed=0).fit(200)\n"
        "a.save('q_no_torch.relepy')\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(pytest.importorskip("pathlib").Path(__file__).parent.parent))
    try:
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "ok"
    finally:
        import pathlib

        pathlib.Path(__file__).parent.parent.joinpath("q_no_torch.relepy").unlink(missing_ok=True)


def test_load_vectorized_agent_with_single_env_instance(tmp_path):
    agent = QLearning(lambda: GridWorldEnv(), n_envs=3, seed=0, verbose=0).fit(300)
    path = tmp_path / "q.relepy"
    agent.save(path)
    loaded = QLearning.load(path, GridWorldEnv(), verbose=0)  # instance -> falls back to 1 env
    assert loaded.n_envs == 1
    np.testing.assert_allclose(loaded.q_table, agent.q_table)
