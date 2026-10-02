import pytest

pytest.importorskip("pyspark")


@pytest.fixture(scope="session")
def spark_session():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("Hypex-Pytest")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        # Deliberately differs from the Spark default (200): with 2 partitions
        # groupBy().collect() returns groups in a different order, which used
        # to swap control/test in ABTest and flip the effect sign. This
        # setting is what lets test_group_order.py and
        # test_abtest_spark_matches_pandas catch such a regression. Do not
        # remove it.
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
