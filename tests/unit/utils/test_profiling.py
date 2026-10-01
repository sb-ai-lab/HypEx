"""Tests for the timeit decorator and profiling switches."""
from __future__ import annotations

import logging

import pytest

from hypex.utils import profiling
from hypex.utils.profiling import (
    ProfilingContext,
    disable_profiling,
    enable_profiling,
    is_profiling_enabled,
    timeit,
)


@pytest.fixture(autouse=True)
def restore_profiling_flag():
    previous = profiling._PROFILING_ENABLED
    yield
    profiling._PROFILING_ENABLED = previous


def test_global_flag_toggles() -> None:
    disable_profiling()
    assert is_profiling_enabled() is False
    enable_profiling()
    assert is_profiling_enabled() is True
    disable_profiling()
    assert is_profiling_enabled() is False


def test_disabled_profiling_is_silent_and_transparent(capsys) -> None:
    disable_profiling()

    @timeit()
    def add(a, b=1):
        """doc"""
        return a + b

    assert add(1, b=2) == 3
    assert capsys.readouterr().out == ""
    assert add.__name__ == "add" and add.__doc__ == "doc"


def test_enabled_profiling_prints_ok_message(capsys) -> None:
    enable_profiling()

    @timeit(level="SPARK", prefix="CALC")
    def work():
        return 42

    assert work() == 42
    out = capsys.readouterr().out
    assert out.startswith("[OK] CALC[SPARK] work: ")
    assert out.rstrip().endswith("s")


def test_explicit_enabled_overrides_global_flag(capsys) -> None:
    disable_profiling()

    @timeit(enabled=True)
    def work():
        return 1

    work()
    assert "[OK]" in capsys.readouterr().out

    enable_profiling()

    @timeit(enabled=False)
    def quiet():
        return 1

    quiet()
    assert capsys.readouterr().out == ""


def test_method_name_includes_class(capsys) -> None:
    class Worker:
        @timeit(level="PROC", enabled=True)
        def run(self):
            return "x"

    assert Worker().run() == "x"
    assert "[PROC] Worker.run" in capsys.readouterr().out


def test_failure_is_reported_and_reraised(capsys) -> None:
    @timeit(enabled=True)
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        boom()
    out = capsys.readouterr().out
    assert "[FAIL] [INFO] boom: FAILED after" in out
    assert "ValueError: bad" in out


@pytest.mark.parametrize("elapsed,marker", [(0.0, "[OK]"), (150.0, "[WARN]"), (400.0, "[SLOW]")])
def test_markers_follow_thresholds(monkeypatch, capsys, elapsed, marker) -> None:
    ticks = iter([0.0, elapsed])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))

    @timeit(enabled=True)
    def work():
        return None

    work()
    assert capsys.readouterr().out.startswith(marker)


def test_threshold_constants() -> None:
    assert profiling.WARN_THRESHOLD == 100.0
    assert profiling.SLOW_THRESHOLD == 300.0


@pytest.mark.parametrize(
    "elapsed,expected_level",
    [(0.0, logging.DEBUG), (150.0, logging.INFO), (400.0, logging.WARNING)],
)
def test_logger_output_levels(monkeypatch, caplog, elapsed, expected_level) -> None:
    ticks = iter([0.0, elapsed])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))

    @timeit(enabled=True, log_to_console=False, log_to_logger=True)
    def work():
        return None

    with caplog.at_level(logging.DEBUG, logger=profiling.logger.name):
        work()
    records = [r for r in caplog.records if r.name == profiling.logger.name]
    assert [r.levelno for r in records] == [expected_level]


def test_log_to_console_can_be_disabled(capsys) -> None:
    @timeit(enabled=True, log_to_console=False)
    def work():
        return None

    work()
    assert capsys.readouterr().out == ""


def test_failure_goes_to_logger_as_error(caplog) -> None:
    @timeit(enabled=True, log_to_console=False, log_to_logger=True)
    def boom():
        raise RuntimeError("x")

    with caplog.at_level(logging.DEBUG, logger=profiling.logger.name):
        with pytest.raises(RuntimeError):
            boom()
    assert any(r.levelno == logging.ERROR and "FAILED" in r.message for r in caplog.records)


def test_profiling_context_enables_and_restores() -> None:
    disable_profiling()
    with ProfilingContext(enabled=True) as ctx:
        assert is_profiling_enabled() is True
        assert ctx.previous_state is False
    assert is_profiling_enabled() is False


def test_profiling_context_can_disable_and_restores_on_error() -> None:
    enable_profiling()
    with pytest.raises(RuntimeError):
        with ProfilingContext(enabled=False):
            assert is_profiling_enabled() is False
            raise RuntimeError("boom")
    assert is_profiling_enabled() is True


def test_nested_contexts_restore_in_order() -> None:
    disable_profiling()
    with ProfilingContext(True):
        with ProfilingContext(False):
            assert is_profiling_enabled() is False
        assert is_profiling_enabled() is True
    assert is_profiling_enabled() is False


@pytest.mark.xfail(
    strict=True,
    reason="Issue: timeit docstring promises [SLOW] > 10 s and [WARN] > 1 s, but the "
    "implementation uses 300 s and 100 s",
)
def test_documented_thresholds_match_implementation() -> None:
    assert profiling.SLOW_THRESHOLD == 10.0 and profiling.WARN_THRESHOLD == 1.0
