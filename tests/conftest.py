"""Root conftest for HypEx test suite.

Provides session-scoped Spark session, numpy seed isolation,
backend parametrization, and dataset factory fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import ABCRole, Dataset
from hypex.utils import BackendsEnum


# ---------------------------------------------------------------------------
# Spark session fixture (session-scoped, reused across all tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark_session():
    """Create a session-scoped SparkSession for Spark-backend tests.

    Configured with minimal settings for fast local execution:
    - local[*] master
    - UI disabled
    - 2 shuffle partitions (avoids empty-partition edge cases)
    - Arrow disabled to avoid serialization surprises in tests

    Yields:
        pyspark.sql.SparkSession: Active Spark session.

    Note:
        Tests that need this fixture should be marked with
        ``@pytest.mark.spark`` so they can be deselected in
        fast developer runs (``-m "not spark"``).
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        pytest.skip("PySpark is not installed")

    session = (
        SparkSession.builder.master("local[*]")
        .appName("HypEx-Tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    # Suppress noisy Spark logs during tests
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


# ---------------------------------------------------------------------------
# Numpy random state isolation (autouse — every test gets a clean RNG)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def numpy_seed_guard():
    """Save and restore numpy random state around every test.

    This prevents test ordering from affecting results when code
    under test calls ``np.random.seed()`` globally (as several
    tutorial data generators do). Each test starts with the RNG
    state that was active before the test and leaves it unchanged
    after the test completes.

    Yields:
        None
    """
    state = np.random.get_state()
    yield
    np.random.set_state(state)


# ---------------------------------------------------------------------------
# Backend parametrization fixture
# ---------------------------------------------------------------------------
@pytest.fixture(params=[BackendsEnum.pandas, BackendsEnum.spark])
def backend(request, spark_session):
    """Parametrize tests over Pandas and Spark backends.

    When the parameter is ``BackendsEnum.spark``, the test is
    automatically marked with ``pytest.mark.spark`` so it can be
    deselected in fast runs.

    Args:
        request: pytest request object providing the parameter value.
        spark_session: Session-scoped Spark session (unused for pandas
            but required to be available when spark is selected).

    Yields:
        BackendsEnum: The backend enum value for this test iteration.
    """
    backend_value = request.param
    if backend_value == BackendsEnum.spark:
        request.node.add_marker(pytest.mark.spark)
    yield backend_value


# ---------------------------------------------------------------------------
# Dataset factory fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def make_dataset(backend, spark_session):
    """Factory fixture for creating Dataset instances on the active backend.

    Builds a ``Dataset`` from a ``pd.DataFrame`` and a roles mapping,
    automatically selecting the backend provided by the ``backend``
    fixture. For Spark backend, the session-scoped ``spark_session``
    is passed through.

    Args:
        backend: The backend enum value from the ``backend`` fixture.
        spark_session: Session-scoped Spark session.

    Returns:
        Callable: A factory function with signature
        ``(data: pd.DataFrame, roles: dict) -> Dataset``.

    Example:
        .. code-block:: python

            def test_something(make_dataset):
                df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
                ds = make_dataset(df, {"x": FeatureRole(), "y": TargetRole()})
                assert len(ds) == 3
    """

    def _factory(
        data: pd.DataFrame,
        roles: dict[str, ABCRole],
        session=None,
    ) -> Dataset:
        """Create a Dataset from a pandas DataFrame and roles mapping.

        Args:
            data: Source pandas DataFrame.
            roles: Column name to role mapping.
            session: Optional Spark session override. If None, uses
                the session-scoped spark_session for spark backend.

        Returns:
            Dataset: A new Dataset instance on the active backend.
        """
        effective_session = session
        if backend == BackendsEnum.spark and effective_session is None:
            effective_session = spark_session

        return Dataset(
            data=data,
            roles=roles,
            backend=backend,
            session=effective_session,
        )

    return _factory


# ---------------------------------------------------------------------------
# Pytest collection hooks — auto-mark spark tests
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    """Automatically add ``spark`` marker to tests using spark_session.

    Any test that requests the ``spark_session`` fixture directly
    (not through the ``backend`` parametrization) gets the ``spark``
    marker so it can be deselected with ``-m "not spark"``.

    Args:
        config: pytest config object.
        items: List of collected test items.
    """
    for item in items:
        if "spark_session" in getattr(item, "fixturenames", []):
            # Only add if not already marked (backend fixture handles its own)
            if not any(marker.name == "spark" for marker in item.iter_markers()):
                item.add_marker(pytest.mark.spark)


# ---------------------------------------------------------------------------
# Per-backend strict xfail helper
# ---------------------------------------------------------------------------
@pytest.fixture
def xfail_backend(request, backend):
    """Mark the running test as a strict xfail for the given backends only.

    Usage inside a test: ``xfail_backend(BackendsEnum.spark, reason="...")``.
    Other backends run the test normally.
    """

    def _apply(*backends, reason, raises=None):
        if backend in backends:
            request.applymarker(
                pytest.mark.xfail(strict=True, reason=reason, raises=raises)
            )

    return _apply
