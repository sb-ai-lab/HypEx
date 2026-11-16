"""Common fixtures for the StorageManager test suite."""

import pytest

from hypex.dataset.storage_manager import StorageManager


@pytest.fixture(autouse=True)
def fresh_manager():
    """
    Each test starts with a clean registry. After the test, the registry is also cleaned:
    all references forgotten by the test are released (unpersist is called).
    """
    StorageManager.reset(release=True)
    yield StorageManager.instance()
    StorageManager.reset(release=True)


@pytest.fixture
def spark_dataset(spark_session):
    """Build real Spark datasets using the suite's session and explicit index."""
    from pyspark.sql import functions as F

    from hypex.dataset import Dataset, FeatureRole, InfoRole

    def build(size=12):
        frame = (
            spark_session.range(size)
            .withColumn("group", F.col("id") % 2)
            .withColumn("value", F.col("id") + 1)
        )
        return Dataset(
            data=frame.pandas_api(index_col="id"),
            roles={"group": InfoRole(), "value": FeatureRole()},
        )

    return build
