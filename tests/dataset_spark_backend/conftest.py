import pytest

pytest.importorskip("pyspark")


@pytest.fixture(scope="session")
def spark_session():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("Hypex-Pytest")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
