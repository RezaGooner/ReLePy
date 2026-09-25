import pytest

from relepy.utils.schedules import constant_schedule, exponential_schedule, linear_schedule
from relepy.utils.logger import Logger


def test_linear_schedule():
    s = linear_schedule(1.0, 0.1, 100)
    assert s(0) == pytest.approx(1.0)
    assert s(50) == pytest.approx(0.55)
    assert s(100) == pytest.approx(0.1)
    assert s(10_000) == pytest.approx(0.1)


def test_other_schedules():
    assert constant_schedule(0.3)(123) == 0.3
    e = exponential_schedule(1.0, 0.0, 10)
    assert e(0) == pytest.approx(1.0)
    assert 0.0 < e(10) < 1.0


def test_logger_history(capsys):
    log = Logger(verbose=0)
    log.record("a", 1.0)
    log.dump(5)
    assert log.history == [{"timesteps": 5, "a": 1.0}]
    assert capsys.readouterr().out == ""
