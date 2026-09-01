"""
HypEx Logging Module.

Provides a flexible and configurable logger (`HypExLogger`) designed for 
machine learning and data processing pipelines. 

Key features:
- Console and file logging with configurable levels.
- Function and method decorators for automatic execution tracing.
- Context managers for pipeline process tracking.
- PySpark session and job monitoring.
- Environment-based configuration (`HYPEX_LOG_LEVEL`, `HYPEX_LOG_FILE`) 
  to control verbosity globally without code changes.
"""
from __future__ import annotations

import functools
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)

# Context variable to store metadata about the currently executing pipeline process.
# This allows nested decorators to prefix their logs with the active process name.
_current_process: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_process", default=None
)


class HypExLogger:
    """A versatile logger with decorator and context manager capabilities.
    
    Supports standard logging, function/method tracing, and pipeline process 
    tracking. By default, it reads configuration from environment variables 
    to prevent excessive console spam in production environments.
    
    Environment Variables:
        HYPEX_LOG_LEVEL: Default logging level (e.g., "DEBUG", "INFO", "WARNING"). 
            Defaults to "WARNING" if not set.
        HYPEX_LOG_FILE: Default path for log file output. If not set, no file 
            handler is created.
            
    Examples:
        Create a logger and use it as a decorator:
        
        >>> logger = HypExLogger(name="my_app", level="INFO")
        >>> @logger
        ... def my_function(x, y):
        ...     return x + y
        
        Use as a context manager for pipeline steps:
        
        >>> with logger.process("DataLoader", backend="spark"):
        ...     load_data()
    """

    def __init__(
        self,
        name: str = "hypex",
        level: str | None = None,
        log_file: str | None = None,
        fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Initializes the HypExLogger.
        
        Args:
            name: The name of the logger.
            level: Logging level (e.g., "DEBUG", "INFO", "WARNING"). 
                If None, reads from `HYPEX_LOG_LEVEL` env var, defaulting to "WARNING".
            log_file: Optional path to a log file. If None, reads from 
                `HYPEX_LOG_FILE` env var. If still None, no file handler is created.
            fmt: Log message format string.
            datefmt: Date format string.
        """
        if level is None:
            level = os.getenv("HYPEX_LOG_LEVEL", "WARNING")
        if log_file is None:
            log_file = os.getenv("HYPEX_LOG_FILE")
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
        self.logger.handlers.clear()
        self.logger.propagate = False
        
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def __call__(
        self,
        func: F | None = None,
        *,
        name: str | None = None,
        log_args: bool = False,
        log_result: bool = False,
    ) -> F | Callable[[F], F]:
        """Decorates a function or method to log its execution.
        
        Can be used with or without arguments:
            @logger
            def foo(): ...
            
            @logger(log_args=True)
            def bar(): ...
            
        Args:
            func: The function to decorate (when used without arguments).
            name: Custom name for the log messages. Defaults to function name.
            log_args: Whether to log function arguments.
            log_result: Whether to log the function's return value.
            
        Returns:
            The decorated function or a decorator function.
        """
        def decorator(f: F) -> F:
            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                process_info = _current_process.get() or {}
                prefix = ""
                if process_info:
                    prefix = f"[{process_info.get('name', '?')}|{process_info.get('backend', '?')}] "
                    
                func_name = name or f.__name__
                
                # Use DEBUG for method tracing to avoid spamming INFO logs
                if log_args:
                    self.logger.debug(f"{prefix}▶ {func_name}(args={args}, kwargs={kwargs})")
                else:
                    self.logger.debug(f"{prefix}▶ {func_name}")
                    
                start = time.perf_counter()
                try:
                    result = f(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    if log_result:
                        self.logger.debug(f"{prefix}✓ {func_name} completed in {elapsed:.3f}s, result={result}")
                    else:
                        self.logger.debug(f"{prefix}✓ {func_name} completed in {elapsed:.3f}s")
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    self.logger.error(
                        f"{prefix}✗ {func_name} failed after {elapsed:.3f}s: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise
            return wrapper  # type: ignore
            
        if func is None:
            return decorator  # type: ignore
        return decorator(func)

    def log_methods(
        self,
        log_args: bool = False,
        log_result: bool = False,
        exclude: list[str] | None = None,
        private: bool = False,
        static: bool = False,
    ) -> Callable[[C], C]:
        """Class decorator that automatically logs calls to its methods.
        
        Args:
            log_args: Whether to log method arguments.
            log_result: Whether to log method return values.
            exclude: List of method names to exclude from logging.
            private: Whether to log private (starting with '_') and dunder methods.
            static: Whether to log `@staticmethod` and `@classmethod` decorated methods.
            
        Returns:
            A class decorator.
        """
        def wrapper(cls: C) -> C:
            exclude_set = set(exclude or [])
            
            for attr_name, attr_value in vars(cls).items():
                if attr_name in exclude_set:
                    continue
                    
                is_callable = callable(attr_value)
                is_static = isinstance(attr_value, staticmethod)
                is_classmethod = isinstance(attr_value, classmethod)
                
                if is_callable or is_static or is_classmethod:
                    is_private = attr_name.startswith("_")
                    is_dunder = attr_name.startswith("__") and attr_name.endswith("__")
                    
                    if (is_private and not private) or (is_dunder and not private):
                        continue
                        
                    if is_static:
                        func_to_wrap = attr_value.__func__
                        wrapped_func = self(func_to_wrap, log_args=log_args, log_result=log_result)
                        setattr(cls, attr_name, staticmethod(wrapped_func))
                    elif is_classmethod:
                        func_to_wrap = attr_value.__func__
                        wrapped_func = self(func_to_wrap, log_args=log_args, log_result=log_result)
                        setattr(cls, attr_name, classmethod(wrapped_func))
                    else:
                        wrapped_func = self(attr_value, log_args=log_args, log_result=log_result)
                        setattr(cls, attr_name, wrapped_func)
            return cls
        return wrapper

    def process(
        self,
        name: str,
        backend: str = "unknown",
        log_spark: bool = False,
        **extra: Any,
    ) -> ProcessContext:
        """Creates a context manager for tracking a pipeline process.
        
        Args:
            name: Name of the process (e.g., executor class name).
            backend: Backend type ("pandas", "spark", etc.).
            log_spark: Whether to log Spark session info on enter/exit.
            **extra: Additional metadata to store in the context.
            
        Returns:
            A `ProcessContext` instance.
        """
        return ProcessContext(self, name, backend, log_spark=log_spark, **extra)

    def log_spark_info(self, spark_session: Any | None = None) -> None:
        """Logs detailed information about the active Spark session.
        
        Args:
            spark_session: A PySpark `SparkSession` instance. If None, 
                attempts to retrieve the active session.
        """
        if spark_session is None:
            try:
                from pyspark.sql import SparkSession
                spark_session = SparkSession.getActiveSession()
            except Exception:
                return
                
        if spark_session is None:
            return
            
        try:
            sc = spark_session.sparkContext
            conf = sc.getConf()
            
            self.logger.debug("=" * 60)
            self.logger.debug("Spark Session Info:")
            self.logger.debug(f"  Master: {conf.get('spark.master', 'N/A')}")
            self.logger.debug(f"  App name: {conf.get('spark.app.name', 'N/A')}")
            self.logger.debug(f"  Driver memory: {conf.get('spark.driver.memory', 'N/A')}")
            self.logger.debug(f"  Executor memory: {conf.get('spark.executor.memory', 'N/A')}")
            self.logger.debug(f"  Executor cores: {conf.get('spark.executor.cores', 'N/A')}")
            self.logger.debug(f"  Executor instances: {sc.defaultParallelism}")
            self.logger.debug(f"  Spark version: {spark_session.version}")
            self.logger.debug("=" * 60)
        except Exception as e:
            self.logger.debug(f"Could not log Spark info: {e}")

    def log_spark_process(self, spark_session: Any | None = None) -> None:
        """Logs information about active Spark jobs and stages.
        
        Args:
            spark_session: A PySpark `SparkSession` instance. If None,
                attempts to retrieve the active session.
        """
        if spark_session is None:
            try:
                from pyspark.sql import SparkSession
                spark_session = SparkSession.getActiveSession()
            except Exception:
                return
                
        if spark_session is None:
            return
            
        try:
            sc = spark_session.sparkContext
            tracker = sc.statusTracker()
            job_ids = tracker.getActiveJobIds()
            
            if not job_ids:
                self.logger.debug("No active Spark jobs")
                return
                
            for job_id in job_ids:
                job_info = tracker.getJobInfo(job_id)
                if job_info:
                    stage_ids = job_info.stageIds()
                    for stage_id in stage_ids:
                        stage_info = tracker.getStageInfo(stage_id)
                        if stage_info:
                            self.logger.debug(
                                f"Spark Job {job_id}, Stage {stage_id}: "
                                f"{stage_info.numTasks()} tasks, "
                                f"{stage_info.numActiveTasks()} active, "
                                f"{stage_info.numFailedTasks()} failed"
                            )
        except Exception as e:
            self.logger.debug(f"Could not log Spark process: {e}")

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level DEBUG."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level INFO."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level WARNING."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level ERROR."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level CRITICAL."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs an exception with level ERROR."""
        self.logger.exception(msg, *args, **kwargs)


class ProcessContext:
    """Context manager for tracking pipeline processes.
    
    Automatically logs the start and end of a process, execution time,
    and optionally tracks Spark session/job information.
    
    Attributes:
        logger: The `HypExLogger` instance to use.
        name: Name of the process.
        backend: Backend type.
        log_spark: Whether to log Spark infos.
        extra: Additional metadata.
    """

    def __init__(
        self,
        logger: HypExLogger,
        name: str,
        backend: str,
        log_spark: bool = False,
        **extra: Any,
    ) -> None:
        self.logger = logger
        self.name = name
        self.backend = backend
        self.log_spark = log_spark
        self.extra = extra
        self._token: Any = None
        self._start_time: float = 0.0

    def __enter__(self) -> ProcessContext:
        process_info = {
            "name": self.name,
            "backend": self.backend,
            **self.extra,
        }
        self._token = _current_process.set(process_info)
        self._start_time = time.perf_counter()
        
        self.logger.info(f"▶ Process started: {self.name} [{self.backend}]")
        
        if self.log_spark:
            self.logger.log_spark_info()
            
        return self

    def __exit__(
        self, 
        exc_type: type[BaseException] | None, 
        exc_val: BaseException | None, 
        exc_tb: Any
    ) -> bool:
        elapsed = time.perf_counter() - self._start_time
        
        if exc_type:
            self.logger.error(
                f"✗ Process failed: {self.name} after {elapsed:.3f}s - "
                f"{exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.info(f"✓ Process finished: {self.name} in {elapsed:.3f}s")
            
        if self.log_spark:
            self.logger.log_spark_process()
            
        _current_process.reset(self._token)
        return False


# Global singleton logger instance.
# By default, uses WARNING level and no log file to prevent console spam 
# and unwanted file creation. Users can override this via environment variables:
#   export HYPEX_LOG_LEVEL=INFO
#   export HYPEX_LOG_FILE=experiment.log
logger = HypExLogger(name="hypex.experiment")