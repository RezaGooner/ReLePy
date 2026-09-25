import pytest

from relepy.core.config import BaseConfig, DeepConfig
from relepy.algorithms.tabular import TabularConfig


def test_defaults_and_validation():
    assert BaseConfig().gamma == 0.99
    with pytest.raises(ValueError):
        BaseConfig(gamma=1.5)
    with pytest.raises(ValueError):
        DeepConfig(learning_rate=0)
    with pytest.raises(ValueError):
        DeepConfig(hidden_sizes=())
    with pytest.raises(ValueError):
        DeepConfig(activation="swish")
    with pytest.raises(ValueError):
        TabularConfig(alpha=0.0)


def test_update_and_unknown_keys():
    cfg = TabularConfig()
    new = cfg.update(alpha=0.5)
    assert new.alpha == 0.5 and cfg.alpha == 0.1
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        cfg.update(alhpa=0.5)


def test_json_roundtrip(tmp_path):
    cfg = DeepConfig(hidden_sizes=(32, 32), learning_rate=1e-3)
    path = tmp_path / "cfg.json"
    cfg.to_json(path)
    loaded = DeepConfig.from_json(path)
    assert loaded == cfg
    assert DeepConfig.from_json(cfg.to_json()) == cfg
