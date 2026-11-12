"""
Integration tests on real Spark against YOUR StorageManager:
check exactly what is shown on the Storage tab in Spark UI.

If pyspark is not installed -- the entire file is automatically skipped.
To run only this file: python3 -m pytest tests/test_spark_integration.py -v
"""

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from hypex.dataset.storage_manager import StorageManager

from .fakes import SparkNavigation


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("storage-manager-tests")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def persistent_rdd_count(spark):
    """Number of persistent RDDs -- same as shown on the Storage tab.
    None if the API is not available in this Spark version."""
    try:
        return spark.sparkContext._jsc.sc().getPersistentRDDs().size()
    except Exception:
        return None


class TestSparkScenarios:
    def test_user_scenario_releases_both_cached_plans(self, spark):
        """Your bug: persist -> groupby/agg -> persist -> action -> unpersist.
        Storage should be empty (both cache markers are removed, RDDs are released)."""
        manager = StorageManager.instance()

        df = spark.range(100_000).withColumn("feat_cat", (F.col("id") % 5).cast("string"))
        plan1 = df

        nav = SparkNavigation(plan1)
        nav.persist()
        nav = nav.groupby("feat_cat").agg(F.sum("id").alias("s"))
        plan2 = nav.data
        nav.persist()

        plan2.count()  # action: materialize both caches -- Storage shows 2 plans
        assert plan1.is_cached and plan2.is_cached

        nav.unpersist()

        assert not plan1.is_cached, "leak: plan1 is still in cache"
        assert not plan2.is_cached, "leak: plan2 is still in cache"
        assert manager.total_entries() == 0
        rdds = persistent_rdd_count(spark)
        if rdds is not None:
            assert rdds == 0, "persistent RDDs remain in Storage"

    def test_action_flow_on_real_spark(self, spark):
        """Rule 2 on real Spark: old marker is removed, new one remains."""
        manager = StorageManager.instance()

        df1 = spark.range(1000)
        nav1 = SparkNavigation(df1)
        nav1.persist()
        assert df1.is_cached

        df2 = spark.range(2000)
        nav2 = SparkNavigation(df2, persist_key=nav1.persist_key)
        nav2.persist(action_triggered=True)

        assert not df1.is_cached, "on action, the old cache marker should be removed"
        assert df2.is_cached
        frames = manager.frames(nav2.persist_key)
        assert len(frames) == 1 and frames[0] is df2

        nav2.unpersist()
        assert not df2.is_cached

    def test_independent_lineages_are_isolated(self, spark):
        df1 = spark.range(10)
        df2 = spark.range(20)
        nav1, nav2 = SparkNavigation(df1), SparkNavigation(df2)

        nav1.persist()
        nav2.persist()
        df1.count()
        df2.count()
        assert df1.is_cached and df2.is_cached

        nav1.unpersist()
        assert not df1.is_cached
        assert df2.is_cached, "unpersist of one branch should not affect the other"

        nav2.unpersist()
        assert not df2.is_cached
