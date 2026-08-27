"""
Класс-логгер HypExLogger с возможностью использования как декоратор.

Поддерживает:
- Логирование в файл и/или консоль
- Использование как декоратор для функций/методов
- Контекстный менеджер для логирования процессов
- Автоматическое логирование времени выполнения, аргументов, результата/ошибки
- Логирование информации о Spark-сессии и текущих Spark-процессах
"""
from __future__ import annotations

import functools
import logging
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

# Контекстная переменная для хранения информации о текущем процессе
_current_process: ContextVar[dict | None] = ContextVar('current_process', default=None)


class HypExLogger:
    """Класс-логгер с возможностью использования как декоратор.

    Args:
        name: Имя логгера (по умолчанию "hypex")
        level: Уровень логирования ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Путь к файлу логов (опционально)
        fmt: Формат сообщений
        datefmt: Формат даты

    Примеры использования:
        # Создание логгера
        logger = HypExLogger(name="hypex", level="INFO", log_file="experiment.log")

        # Как декоратор (без аргументов)
        @logger
        def my_function(x, y):
            return x + y

        # Как декоратор (с аргументами)
        @logger(name="custom", log_args=True)
        def my_function(x, y):
            return x + y

        # Как контекстный менеджер для процесса
        with logger.process("DummyEncoder", backend="spark"):
            # код
            pass

        # Как обычный логгер
        logger.info("Сообщение")
        logger.error("Ошибка")

        # Логирование информации о Spark-сессии
        logger.log_spark_info()

        # Логирование текущих Spark-процессов (jobs, stages)
        logger.log_spark_process()
    """

    def __init__(
        self,
        name: str = "hypex",
        level: str = "INFO",
        log_file: str | None = None,
        console_out: bool = True,
        fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()
        self.logger.propagate = False

        formatter = logging.Formatter(fmt, datefmt=datefmt)

        if console_out:
            # Консольный handler
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            self.logger.addHandler(console)

        # Файловый handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        

    def __call__(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        log_args: bool = False,
        log_result: bool = False,
    ) -> Callable | Any:
        """Использовать как декоратор для функции/метода.

        Args:
            func: Функция для декорирования (если вызван без аргументов)
            name: Кастомное имя для логирования (по умолчанию имя функции)
            log_args: Логировать ли аргументы функции
            log_result: Логировать ли результат функции

        Returns:
            Декорированная функция или декоратор
        """
        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # Получаем информацию о текущем процессе
                process_info = _current_process.get() or {}
                prefix = ""
                if process_info:
                    prefix = f"[{process_info.get('name', '?')}|{process_info.get('backend', '?')}] "

                func_name = name or f.__name__

                # Логируем начало
                if log_args:
                    self.logger.info(f"{prefix}▶ {func_name}(args={args}, kwargs={kwargs})")
                else:
                    self.logger.info(f"{prefix}▶ {func_name}")

                start = time.perf_counter()

                try:
                    result = f(*args, **kwargs)
                    elapsed = time.perf_counter() - start

                    if log_result:
                        self.logger.info(f"{prefix}✓ {func_name} completed in {elapsed:.3f}s, result={result}")
                    else:
                        self.logger.info(f"{prefix}✓ {func_name} completed in {elapsed:.3f}s")

                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    self.logger.error(
                        f"{prefix}✗ {func_name} failed after {elapsed:.3f}s: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise

            return wrapper

        if func is None:
            return decorator
        return decorator(func)

    def log_methods(
        self,
        log_args: bool = True,
        log_result: bool = False,
        exclude: list[str] | None = None,
        private: bool = False,
        static: bool = False,  # <-- Новая опция для статических методов и методов класса
    ) -> Callable:
        """Декоратор для класса, который автоматически логирует вызовы его методов.

        Args:
            log_args: Логировать ли аргументы методов
            log_result: Логировать ли результаты методов
            exclude: Список имен методов, которые нужно исключить из логирования
            private: Логгировать ли приватные методы (начинающиеся с '_')
            static: Логгировать ли методы, декорированные как @staticmethod и @classmethod

        Returns:
            Декоратор для класса
        """
        def wrapper(cls):
            exclude_set = set(exclude or [])

            for attr_name, attr_value in vars(cls).items():
                if attr_name in exclude_set:
                    continue

                # Проверяем, является ли атрибут вызываемым или дескриптором метода
                is_callable = callable(attr_value)
                is_static = isinstance(attr_value, staticmethod)
                is_classmethod = isinstance(attr_value, classmethod)

                if is_callable or is_static or is_classmethod:
                    is_private = attr_name.startswith('_')

                    # Если метод приватный, но флаг private=False, пропускаем его
                    # (Также рекомендуется пропускать магические методы __dunder__,
                    # чтобы не логировать __init__, __str__ и т.д., если это не нужно)
                    is_dunder = attr_name.startswith('__') and attr_name.endswith('__')
                    if (is_private and not private) or (is_dunder and not private):
                        continue

                    # Извлекаем саму функцию для корректного обертывания
                    if is_static:
                        func_to_wrap = attr_value.__func__
                        wrapped_func = self(func_to_wrap, log_args=log_args, log_result=log_result)
                        setattr(cls, attr_name, staticmethod(wrapped_func))

                    elif is_classmethod:
                        func_to_wrap = attr_value.__func__
                        wrapped_func = self(func_to_wrap, log_args=log_args, log_result=log_result)
                        setattr(cls, attr_name, classmethod(wrapped_func))

                    else:
                        # Обычный метод или функция
                        wrapped_func = self(attr_value, log_args=log_args, log_result=log_result)
                        setattr(cls, attr_name, wrapped_func)

            return cls
        return wrapper

    def process(
        self,
        name: str,
        backend: str = "unknown",
        log_spark: bool = False,
        **extra
    ) -> ProcessContext:
        """Создать контекстный менеджер для логирования процесса.

        Args:
            name: Имя процесса (например, имя экзекьютора)
            backend: Тип бэкенда ("pandas", "spark", "unknown")
            log_spark: Логировать ли информацию о Spark-сессии
            **extra: Дополнительная информация для логирования

        Returns:
            ProcessContext: Контекстный менеджер
        """
        return ProcessContext(self, name, backend, log_spark=log_spark, **extra)

    def log_spark_info(self, spark_session=None):
        """Логировать информацию о Spark-сессии.

        Args:
            spark_session: SparkSession (если None, используется активная сессия)
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

            self.logger.info("=" * 60)
            self.logger.info("Spark Session Info:")
            self.logger.info(f"  Master: {conf.get('spark.master', 'N/A')}")
            self.logger.info(f"  App name: {conf.get('spark.app.name', 'N/A')}")
            self.logger.info(f"  Driver memory: {conf.get('spark.driver.memory', 'N/A')}")
            self.logger.info(f"  Executor memory: {conf.get('spark.executor.memory', 'N/A')}")
            self.logger.info(f"  Executor cores: {conf.get('spark.executor.cores', 'N/A')}")
            self.logger.info(f"  Executor instances: {sc.defaultParallelism}")
            self.logger.info(f"  Spark version: {spark_session.version}")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.debug(f"Could not log Spark info: {e}")

    def log_spark_process(self, spark_session=None):
        """Логировать информацию о текущем Spark-процессе (jobs, stages).
        
        Args:
            spark_session: SparkSession (если None, используется активная сессия)
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
                            self.logger.info(
                                f"Spark Job {job_id}, Stage {stage_id}: "
                                f"{stage_info.numTasks()} tasks, "
                                f"{stage_info.numActiveTasks()} active, "
                                f"{stage_info.numFailedTasks()} failed"
                            )
        except Exception as e:
            self.logger.debug(f"Could not log Spark process: {e}")

    # Прокси-методы для стандартного логгера
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)


class ProcessContext:
    """Контекстный менеджер для логирования процесса.
    Автоматически логирует начало и конец процесса, время выполнения,
    а также информацию о Spark-сессии и текущих Spark-процессах (если указано).
    """

    def __init__(
        self,
        logger: HypExLogger,
        name: str,
        backend: str,
        log_spark: bool = False,
        **extra
    ):
        self.logger = logger
        self.name = name
        self.backend = backend
        self.log_spark = log_spark
        self.extra = extra
        self._token = None
        self._start_time = None

    def __enter__(self):
        process_info = {
            "name": self.name,
            "backend": self.backend,
            **self.extra
        }
        self._token = _current_process.set(process_info)
        self._start_time = time.perf_counter()

        self.logger.info(f"▶ Process started: {self.name} [{self.backend}]")

        if self.log_spark:
            self.logger.log_spark_info()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

logger = HypExLogger(
    name="hypex.experiment",
    level="INFO",
    log_file="experiment.log",
    console_out=True
)
