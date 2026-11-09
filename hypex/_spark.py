"""Internal compatibility layer for the optional PySpark dependency."""

from __future__ import annotations

PYSPARK_INSTALL_MESSAGE = (
    'Spark support requires PySpark. Install it with pip install "hypex[spark]".'
)

try:
    import pyspark.pandas as ps
    import pyspark.sql as spark
    import pyspark.sql.functions as F
    from pyspark import RDD, Broadcast, SparkConf, SparkFiles, StorageLevel
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.pandas.exceptions import PandasNotImplementedError
    from pyspark.sql import DataFrame as SparkDF
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        ByteType,
        DataType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
except ModuleNotFoundError as exc:
    if exc.name is None or not (
        exc.name == "pyspark" or exc.name.startswith("pyspark.")
    ):
        raise

    PYSPARK_AVAILABLE = False
    ps = None
    spark = None
    F = None
    RDD = None
    Broadcast = None
    SparkConf = None
    SparkFiles = None
    StorageLevel = None
    StringIndexer = None
    VectorAssembler = None
    LinearRegression = None
    PandasNotImplementedError = NotImplementedError
    SparkDF = None
    SparkSession = None
    ArrayType = None
    BooleanType = None
    ByteType = None
    DataType = None
    DateType = None
    DecimalType = None
    DoubleType = None
    FloatType = None
    IntegerType = None
    LongType = None
    ShortType = None
    StringType = None
    StructField = None
    StructType = None
    TimestampType = None
    SPARK_SQL_DATAFRAME_TYPES = ()
    PANDAS_ON_SPARK_DATAFRAME_TYPES = ()
    SPARK_SERIES_TYPES = ()
else:
    PYSPARK_AVAILABLE = True
    SPARK_SQL_DATAFRAME_TYPES = (SparkDF,)
    PANDAS_ON_SPARK_DATAFRAME_TYPES = (ps.DataFrame,)
    SPARK_SERIES_TYPES = (ps.Series,)

SPARK_DATAFRAME_TYPES = SPARK_SQL_DATAFRAME_TYPES + PANDAS_ON_SPARK_DATAFRAME_TYPES


def require_pyspark() -> None:
    if not PYSPARK_AVAILABLE:
        raise ImportError(PYSPARK_INSTALL_MESSAGE)
