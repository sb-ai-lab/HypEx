"""Tests for HypExLogger, ProcessContext and the log_methods class decorator."""

from __future__ import annotations

import logging

import pytest

from hypex.utils.logger import HypExLogger, ProcessContext, logger


@pytest.fixture
def make_logger(request):
    created = []

    def _factory(level="DEBUG", **kwargs):
        lg = HypExLogger(
            name=f"hypex.test.{request.node.name}.{len(created)}", level=level, **kwargs
        )
        created.append(lg)
        return lg

    yield _factory
    for lg in created:
        lg.logger.handlers.clear()


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _capture(lg: HypExLogger) -> _ListHandler:
    handler = _ListHandler()
    lg.logger.addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_default_level_is_warning(monkeypatch, make_logger) -> None:
    monkeypatch.delenv("HYPEX_LOG_LEVEL", raising=False)
    lg = HypExLogger(name="hypex.test.default_level")
    assert lg.logger.level == logging.WARNING
    lg.logger.handlers.clear()


def test_level_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HYPEX_LOG_LEVEL", "debug")
    lg = HypExLogger(name="hypex.test.env_level")
    assert lg.logger.level == logging.DEBUG
    lg.logger.handlers.clear()


def test_unknown_level_falls_back_to_warning(make_logger) -> None:
    assert make_logger(level="nonsense").logger.level == logging.WARNING


def test_explicit_level_wins_over_environment(monkeypatch, make_logger) -> None:
    monkeypatch.setenv("HYPEX_LOG_LEVEL", "ERROR")
    assert make_logger(level="INFO").logger.level == logging.INFO


def test_construction_resets_handlers_and_disables_propagation(make_logger) -> None:
    lg = make_logger()
    assert lg.logger.propagate is False
    assert len(lg.logger.handlers) == 1
    again = HypExLogger(name=lg.logger.name, level="INFO")
    assert len(again.logger.handlers) == 1  # no duplicates when re-created
    again.logger.handlers.clear()


def test_log_file_is_created_and_written(tmp_path, make_logger) -> None:
    path = tmp_path / "nested" / "run.log"
    lg = make_logger(log_file=str(path))
    lg.info("hello file")
    for handler in lg.logger.handlers:
        handler.flush()
    assert "hello file" in path.read_text(encoding="utf-8")


def test_log_file_from_environment(monkeypatch, tmp_path) -> None:
    path = tmp_path / "env.log"
    monkeypatch.setenv("HYPEX_LOG_FILE", str(path))
    lg = HypExLogger(name="hypex.test.env_file", level="INFO")
    lg.info("from env")
    for handler in lg.logger.handlers:
        handler.flush()
    assert "from env" in path.read_text(encoding="utf-8")
    lg.logger.handlers.clear()


@pytest.mark.parametrize(
    "method,level",
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_level_methods(make_logger, method, level) -> None:
    lg = make_logger()
    handler = _capture(lg)
    getattr(lg, method)("msg %s", "x")
    assert [(r.levelno, r.getMessage()) for r in handler.records] == [(level, "msg x")]


def test_exception_logs_traceback(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)
    try:
        raise ValueError("oops")
    except ValueError:
        lg.exception("failed")
    assert handler.records[0].levelno == logging.ERROR
    assert handler.records[0].exc_info is not None


def test_messages_below_level_are_dropped(make_logger) -> None:
    lg = make_logger(level="ERROR")
    handler = _capture(lg)
    lg.info("ignored")
    lg.error("kept")
    assert [r.getMessage() for r in handler.records] == ["kept"]


# ---------------------------------------------------------------------------
# __call__ decorator
# ---------------------------------------------------------------------------
def test_decorator_without_arguments_preserves_behavior(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg
    def add(x, y):
        """Adds."""
        return x + y

    assert add(1, 2) == 3
    assert add.__name__ == "add" and add.__doc__ == "Adds."
    messages = [r.getMessage() for r in handler.records]
    assert any("▶ add" in m for m in messages) and any(
        "✓ add completed" in m for m in messages
    )


def test_decorator_with_arguments_logs_args_and_result(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg(name="custom", log_args=True, log_result=True)
    def mul(x, y=2):
        return x * y

    assert mul(3, y=4) == 12
    messages = [r.getMessage() for r in handler.records]
    assert any("custom(args=(3,), kwargs={'y': 4})" in m for m in messages)
    assert any("result=12" in m for m in messages)


def test_decorator_logs_failure_at_error_level_and_reraises(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg
    def boom():
        raise KeyError("k")

    with pytest.raises(KeyError):
        boom()
    errors = [r for r in handler.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "boom failed" in errors[0].getMessage()


# ---------------------------------------------------------------------------
# log_methods
# ---------------------------------------------------------------------------
def test_log_methods_wraps_public_methods_and_skips_private_by_default(
    make_logger,
) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg.log_methods()
    class Thing:
        def public(self):
            return 1

        def _private(self):
            return 2

        @staticmethod
        def static():
            return 3

    thing = Thing()
    assert (thing.public(), thing._private(), Thing.static()) == (1, 2, 3)
    names = " ".join(r.getMessage() for r in handler.records)
    assert "public" in names
    assert "_private" not in names


@pytest.mark.xfail(
    strict=True,
    reason="Issue: log_methods(static=False) is documented to skip static/class methods, "
    "but the flag is never consulted",
)
def test_log_methods_static_flag_off_skips_static_methods(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg.log_methods()
    class Thing:
        @staticmethod
        def static():
            return 3

    Thing.static()
    assert not any("static" in r.getMessage() for r in handler.records)


def test_log_methods_private_and_static_flags(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg.log_methods(private=True, static=True)
    class Thing:
        def _private(self):
            return 2

        @staticmethod
        def static():
            return 3

        @classmethod
        def klass(cls):
            return 4

    thing = Thing()
    assert (thing._private(), Thing.static(), Thing.klass()) == (2, 3, 4)
    names = " ".join(r.getMessage() for r in handler.records)
    assert "_private" in names and "static" in names and "klass" in names


def test_log_methods_exclude(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg.log_methods(exclude=["skipped"])
    class Thing:
        def skipped(self):
            return 1

        def logged(self):
            return 2

    Thing().skipped()
    Thing().logged()
    names = " ".join(r.getMessage() for r in handler.records)
    assert "logged" in names and "skipped" not in names


# ---------------------------------------------------------------------------
# process / ProcessContext
# ---------------------------------------------------------------------------
def test_process_returns_context(make_logger) -> None:
    lg = make_logger()
    ctx = lg.process("Step", backend="spark", extra_key=1)
    assert isinstance(ctx, ProcessContext)
    assert (ctx.name, ctx.backend, ctx.extra) == ("Step", "spark", {"extra_key": 1})


def test_process_context_logs_start_and_finish(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)
    with lg.process("Step", backend="pandas"):
        pass
    messages = [r.getMessage() for r in handler.records]
    assert any("Process started: Step [pandas]" in m for m in messages)
    assert any("Process finished: Step in" in m for m in messages)


def test_process_context_logs_failure_and_does_not_swallow(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)
    with pytest.raises(ValueError):
        with lg.process("Step"):
            raise ValueError("bad")
    errors = [r.getMessage() for r in handler.records if r.levelno == logging.ERROR]
    assert (
        errors
        and "Process failed: Step" in errors[0]
        and "ValueError: bad" in errors[0]
    )


def test_decorated_calls_inside_process_get_a_prefix(make_logger) -> None:
    lg = make_logger()
    handler = _capture(lg)

    @lg
    def inner():
        return 1

    with lg.process("Outer", backend="spark"):
        inner()
    inner()
    prefixed = [r.getMessage() for r in handler.records if "▶ inner" in r.getMessage()]
    assert prefixed[0].startswith("[Outer|spark] ")
    assert not prefixed[1].startswith("[")


def test_process_context_resets_after_exit(make_logger) -> None:
    from hypex.utils.logger import _current_process

    lg = make_logger()
    assert _current_process.get() in (None, {}) or _current_process.get() == {}
    with lg.process("A"):
        assert _current_process.get()["name"] == "A"
    assert not _current_process.get()


def test_log_spark_helpers_do_not_raise_without_session(make_logger) -> None:
    lg = make_logger()
    lg.log_spark_info(spark_session=object())  # broken session -> swallowed
    lg.log_spark_process(spark_session=object())


@pytest.mark.spark
def test_log_spark_info_with_real_session(make_logger, spark_session) -> None:
    lg = make_logger()
    handler = _capture(lg)
    lg.log_spark_info(spark_session)
    lg.log_spark_process(spark_session)
    messages = " ".join(r.getMessage() for r in handler.records)
    assert "Spark Session Info" in messages and "Spark version" in messages


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="Issue: log_spark_process calls StatusTracker.getActiveJobIds, which PySpark does "
    "not provide, so the error is swallowed and job info is never logged",
)
def test_log_spark_process_reports_job_status(make_logger, spark_session) -> None:
    lg = make_logger()
    handler = _capture(lg)
    lg.log_spark_process(spark_session)
    messages = " ".join(r.getMessage() for r in handler.records)
    assert "Could not log Spark process" not in messages


def test_module_singleton_is_a_hypex_logger() -> None:
    assert isinstance(logger, HypExLogger)
    assert logger.logger.name == "hypex.experiment"
