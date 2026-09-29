import os

import pytest

# pyspark.pandas emits a warning unless this is set before it is imported;
# hypex imports pyspark.pandas eagerly, and conftest.py is loaded before any
# test module, so setting it here is early enough.
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")


@pytest.fixture(scope="session")
def spark_session():
    """A small local SparkSession shared by the whole test session.

    Skips instead of failing where a JVM or pyspark is unavailable, so the CI
    matrix (py3.8-3.13 x linux/macos/windows) cannot go red on it.
    """
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("Hypex-Pytest")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"local SparkSession unavailable: {exc}")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
