import os
import sys

import pytest

pytest.importorskip("pyspark")


@pytest.fixture(scope="session")
def spark_session():
    # Spark launches Python workers by name. Without this they pick up whatever
    # `python` is first on PATH -- not the interpreter (and virtualenv) running
    # the tests -- and crash on the missing pandas/pyarrow, most visibly on
    # Windows CI.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
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
