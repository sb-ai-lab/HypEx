from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Iterable, Sequence, Sized
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyspark.sql as spark
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import (
    DataFrame as SparkDF,
)
from pyspark.sql import (
    Row,
    SparkSession,
    Window,
)
from pyspark.sql import (
    functions as F,
)
from pyspark.sql import (
    types as T,
)
from pyspark.sql.functions import lit, monotonically_increasing_id
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    NumericType,
    StringType,
    StructField,
    StructType,
)

from ...utils import (
    UTILITY_COL_SYMBOL,
    Adapter,
    FromDictTypes,
    MergeOnError,
    ScalarType,
)
from ...utils.constants import UTILITY_INDEX_COL_NAME
from ...utils.typings import PysparkScalarType
from ...utils.typings import SparkTypeMapper as d
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(
        filename: str | Path, session: SparkSession
    ) -> spark.DataFrame:
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return (
                session.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "true")  # TODO: find faster solution in future
                .load(filename)
            )
        elif file_extension == ".parquet":
            return session.read.parquet(filename)
        else:
            try:
                return session.read.table(filename)
            except:
                raise ValueError(f"Unsupported file extension {file_extension}")

    @staticmethod
    def _get_spark_session(
        app_name: str = "HypEx",
        python_path: str | None = None,
        dynamic_allocation: bool = True,
        mode: str | None = None,
    ):
        if python_path is None:
            python_path = sys.executable

        os.environ["PYSPARK_PYTHON"] = python_path
        os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

        if mode == "local":
            conf = (
                SparkConf()
                .setAppName(app_name)
                .setMaster("local[*]")
                .set("spark.driver.memory", "6g")
                .set("spark.executor.memory", "6g")
            )
        else:
            conf = (
                SparkConf()
                .setAppName(app_name)
                .
                # setMaster("yarn").
                set("spark.executor.cores", "8")
                .set("spark.executor.memory", "8g")
                .set("spark.executor.memoryOverhead", "8g")
                .set("spark.driver.cores", "12")
                .set("spark.driver.memory", "16g")
                .set("spark.driver.maxResultSize", "32g")
                .set("spark.shuffle.service.enabled", "true")
                .set("spark.dynamicAllocation.enabled", dynamic_allocation)
                .set("spark.dynamicAllocation.initialExecutors", "6")
                .set("spark.dynamicAllocation.maxExecutors", "32")
                .set("spark.dynamicAllocation.executorIdleTimeout", "120s")
                .set("spark.dynamicAllocation.cachedExecutorIdleTimeout", "600s")
                .set("spark.port.maxRetries", "150")
            )

        return SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

    def __init__(
        self,
        data: spark.DataFrame | pd.DataFrame | dict | str | None = None,
        session: SparkSession = None,
    ):
        if session is None:
            if isinstance(data, spark.DataFrame):
                self.session = data.sparkSession
            else:
                self.session = self._get_spark_session()
        else:
            if isinstance(session, SparkSession):
                self.session = session
            else:
                raise TypeError("Session must be an instance of SparkSession")

        if isinstance(data, dict):
            if "index" in data.keys():
                data = pd.DataFrame(data=data["data"], index=data["index"])
            else:
                data = pd.DataFrame(data=data["data"])

        if isinstance(data, spark.DataFrame):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.data = self.session.createDataFrame(data)
        elif isinstance(data, str):
            self.data = self._read_file(data, self.session)
        else:
            self.data = self.session.createDataFrame([], StructType([]))

    def _add_row_index(self, df: spark.DataFrame, index_column_name: str):
        rdd = df.rdd.zipWithIndex()
        # new_rdd = rdd.map(lambda x: (x[1],) + tuple(x[0]))
        new_rdd = rdd.map(lambda x: (int(x[1]),) + tuple(x[0]))
        new_schema = StructType(
            [StructField(index_column_name, LongType(), False)] + list(self.data.schema.fields)
        )
        return self.session.createDataFrame(new_rdd, new_schema)

    def __getitem__(self, item):
        raise NotImplementedError("Spark-base Dataset does not support indexing")

    def __len__(self):
        return 0 if self.data is None else self.data.count()

    @staticmethod
    def __magic_determine_other(other) -> Any:
        if isinstance(other, SparkDataset):
            return other.data
        elif isinstance(other, spark.DataFrame):
            return other
        elif isinstance(other, F.Column):
            return other
        elif isinstance(other, (int, float, str, bool)):
            return F.lit(other)
        elif isinstance(other, (np.integer, np.floating, np.bool_)):
            return F.lit(other.item())
        else:
            raise TypeError(f"Unsupported operand type: '{type(other).__name__}'. ")

    # comparison operators:
    def __eq__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) == other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __ne__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) != other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __lt__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) < other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __le__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) <= other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __ge__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) >= other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __gt__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) > other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    # Unary operations:
    def __pos__(self) -> spark.DataFrame:
        new_df = self.data.select("*")
        return self.add_column(new_df)

    def __neg__(self) -> spark.DataFrame:
        new_df = self.data.select((-F.col(c)).alias(c) for c in self.data.columns)
        return self.add_column(new_df)

    def __abs__(self) -> spark.DataFrame:
        new_df = self.data.select(F.abs(F.col(c)).alias(c) for c in self.data.columns)
        return self.add_column(new_df)

    def __invert__(self) -> spark.DataFrame:
        new_df = self.data.select((~F.col(c)).alias(c) for c in self.data.columns)
        return self.add_column(new_df)

    def __round__(self, ndigits: int = 0) -> spark.DataFrame:
        new_df = self.data.select(
            F.round(F.col(c), ndigits).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    # Binary operations:
    def __add__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) + other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __sub__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) - other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __mul__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) * other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __floordiv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.floor(F.col(c) / other)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __truediv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) / other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __div__(self, other) -> spark.DataFrame:
        return self.__truediv__(other)

    def __mod__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) % other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __pow__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            F.pow(F.col(c), other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __and__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) & other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __or__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) | other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __truediv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.col(c) / other).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __div__(self, other) -> spark.DataFrame:
        return self.__truediv__(other)

    # Right arithmetic operators:
    def __radd__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (other + F.col(c)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __rsub__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (other - F.col(c)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __rmul__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (other * F.col(c)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __rfloordiv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (F.floor(other / F.col(c))).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __rtruediv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (other / F.col(c)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __rdiv__(self, other) -> spark.DataFrame:
        return self.__rtruediv__(other)

    def __rmod__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            (other % F.col(c)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __rpow__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        new_df = self.data.select(
            F.pow(other, F.col(c)).alias(c) for c in self.data.columns
        )
        return self.add_column(new_df)

    def __repr__(self) -> str:
        df: SparkDF = self.data
        cols = df.columns
        total = df.count()
        if total <= 10:
            pandas_df = df.toPandas()
        else:
            head_df = df.limit(5).toPandas()
            tail_rows = df.tail(5)
            tail_df = pd.DataFrame(tail_rows, columns=cols)
            ellipsis = pd.DataFrame([["..."] * len(cols)], columns=cols)
            pandas_df = pd.concat([head_df, ellipsis, tail_df], ignore_index=True)
        table = pandas_df.__repr__()
        dim = f"\n\n[{total} rows × {len(cols)} columns]"
        return table + dim

    def _repr_html_(self):
        return self.data._repr_html_()

    def get(
        self,
        key,
        default=None,
    ) -> Any:
        key = Adapter.to_list(key)
        # Case 1: key is a string (column name)
        if isinstance(key[0], str):
            columns = [k for k in key if k in self.columns]
            if columns:
                return self.data.select(*columns)
            else:
                return default

        # Case 2: key is an integer (row index)
        elif isinstance(key[0], int):
            df_with_index = self.data.withColumn(
                "__row_index__",
                F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
                - 1,
            )
            return df_with_index.filter(F.col("__row_index__") == key).drop(
                "__row_index__"
            )

        # Case 3: key is a tuple (row_index, column_name)
        elif isinstance(key[0], tuple) and len(key[0]) == 2:
            row_idx, col_name = key[0]
            return self.data.select(col_name).limit(row_idx + 1).tail(1)

        else:
            return default

    def take(
        self,
        indices: int | list[int] | slice,
        axis: Literal["index", "columns", "rows"] | int = 0,
    ) -> Any:
        if isinstance(indices, slice):
            return self.data.offset(indices.start).limit(indices.stop - indices.start + 1)
        indices = Adapter.to_list(indices)
        if axis == 1:
            indices = [self.columns[index] for index in indices]
        return self.get(indices)

    def get_values(
        self,
        row: str | None = None,
        column: str | None = None,
    ) -> Any:
        if row:
            try:
                row = int(row)
            except:
                raise TypeError("For spark-based Dataset row can be accessed by its index only")
        if (column is not None) and (row is not None):
            return self.data.take(row+1)[row][column]
        if column is not None:
            return self.data.select(column).toPandas.values.tolist()
        if row is not None:
            return self.data.take(row+1)[row].toPandas.values.tolist()
        return self

    def iget_values(
        self,
        row: int | None = None,
        column: int | None = None,
    ) -> Any:
        if (column is not None) and (row is not None):
            column = self.columns[column]
            return self.data.take(row+1)[row][column]
        if column is not None:
            column = self.columns[column]
            self.data.select(column).toPandas.values.tolist()
        if row is not None:
            return self.data.take(row+1)[row].toPandas.values.tolist()
        return self

    def create_empty(
        self,
        index: Iterable | None | spark.DataFrame = None,
        columns: Iterable[str] | None = None,
    ):
        columns = list(columns or [])
        schema = T.StructType(
            [T.StructField(column, T.StringType(), True) for column in columns]
        )
        empty_rdd = self.session.sparkContext.emptyRDD()
        self.data = self.session.createDataFrame(empty_rdd, schema)
        if index is not None:
            if isinstance(index, spark.DataFrame):
                index_df = index.select(UTILITY_INDEX_COL_NAME)
            else:
                index_rows = [(value,) for value in index]
                index_df = self.session.createDataFrame(index_rows, ["index"])
            self.data = index_df.join(self.data, how="cross") if columns else index_df
        return self

    @property
    def index(self):
        if self.data is None:
            return []
        if UTILITY_INDEX_COL_NAME in self.data.columns:
            return self.data.select(UTILITY_INDEX_COL_NAME)
        return self.add_index_col()

    @property
    def columns(self):
        return self.data.columns if self.data else []

    # @property
    # def session(self):
    #     return self.session

    @property
    def shape(self):
        if self.data:
            count = self.data.count()
            cols = len(self.data.columns)
            return (count, cols)
        return (0, 0)

    def _get_column_index(
        self, column_name: Sequence[str] | str
    ) -> int | Sequence[int]:
        pd_index_columns = pd.Index(self.data.columns)
        if isinstance(column_name, str):
            return pd_index_columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return pd_index_columns.get_indexer(column_name)
        else:
            raise TypeError("Wrong column_name type.")

    def get_column_type(
        self, column_name: list[str] | str = None
    ) -> dict[str, type] | type | None:
        column_name = self.data.columns if column_name is None else column_name
        dtypes = {}
        for k, v in self.data.select(column_name).dtypes:
            if pd.api.types.is_integer_dtype(v) or v == "bigint":
                dtypes[k] = int
            elif pd.api.types.is_float_dtype(v):
                dtypes[k] = float
            elif (
                pd.api.types.is_string_dtype(v)
                or pd.api.types.is_object_dtype(v)
                or v == "category"
            ):
                dtypes[k] = str
            elif pd.api.types.is_bool_dtype(v):
                dtypes[k] = bool
        if isinstance(column_name, list):
            return dtypes
        else:
            if column_name in dtypes:
                return dtypes[column_name]
        return None

    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> spark.DataFrame:
        for col, new_type in dtype.items():
            new_type = str(new_type.__name__)
            self.data = self.data.withColumn(col, self.data[col].cast(new_type))
        return self.data

    def update_column_type(self, dtype: dict[str, type]):
        if len(dtype) > 0:
            self.data = self.astype(dtype)
        return self

    def add_column(
        self, data: Any, name: str | None = None, index=None
    ) -> spark.DataFrame:
        def _add_columns_from_dataframe(
            df: spark.DataFrame, new_df: spark.DataFrame
        ) -> spark.DataFrame:
            if df.count() != new_df.count():
                raise ValueError(
                    f"Row count mismatch: original DF has {df.count()} rows, new DF has {new_df.count()} rows"
                )

            df_with_index = df.withColumn("__join_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1)
            new_df_with_index = new_df.withColumn(
                "__join_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1
            )

            result = df_with_index.join(new_df_with_index, "__join_id", "inner")

            self.data = result.drop("__join_id")

        def _add_columns_from_list(
            df: spark.DataFrame, data_list: list, column_names: str
        ) -> spark.DataFrame:
            if len(data_list) != df.count():
                raise ValueError(
                    f"Data length {len(data_list)} doesn't match DataFrame row count {df.count()}"
                )

            original_rdd = df.rdd
            zipped_rdd = original_rdd.zip(
                self.session.sparkContext.parallelize(data_list)
            )
            new_df = zipped_rdd.map(lambda x: x[0] + (x[1],)).toDF(
                df.columns + column_names
            )

            self.data = new_df

        if isinstance(data, spark.DataFrame):
            return _add_columns_from_dataframe(self.data, data)
        elif isinstance(data, list):
            return _add_columns_from_list(self.data, data, name)
        elif not isinstance(data, Iterable):
            self.data = self.data.withColumn(name, F.lit(data))
            return
        else:
            raise ValueError(
                "new_data must be Spark DataFrame, list of values, or list of lists"
            )

    def append(  # Эрик
        self, other, reset_index: bool = False, axis: int = 0 # other: Union["SparkDataset", List["SparkDataset"]]
    ) -> spark.DataFrame:
        datasets = [self.data] + [d.data for d in other]
        if axis == 0:
            new_data = reduce(lambda x, y: x.unionByName(y, allowMissingColumns=True), datasets)
            if reset_index:
                return new_data
            return new_data
        if axis == 1:
            base = datasets[0].withColumn("__row_id", F.monotonically_increasing_id())
            for dataset in datasets[1:]:
                right = dataset.withColumn("__row_id", F.monotonically_increasing_id())
                base = base.join(right, on="__row_id", how="inner")
            return base.drop("__row_id")
        raise ValueError("axis should be 0 or 1")

    def from_dict(
        self, data: FromDictTypes, index: Iterable | Sized | None = None
    ):
        spark_session = self.session
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            row_count = len(next(iter(data.values())))
            index_list = list(index) if index is not None else list(range(row_count))
            rows = [
                {"_index": idx, **{k: v[idx] for k, v in data.items()}}
                for idx in index_list
            ]
            self.data = spark_session.createDataFrame(rows)
        elif isinstance(data, list) and all(isinstance(row, dict) for row in data):
            if index is not None:
                rows = [{**row, "_index": idx} for idx, row in zip(index, data)]
            else:
                rows = [{**row, "_index": i} for i, row in enumerate(data)]
            self.data = spark_session.createDataFrame(rows)
        else:
            raise ValueError("Unsupported data format for from_dict")
        return self

    def to_dict(self) -> dict[str, Any]:  # TODO: to be moved to small data
        rows = self.data.collect()
        return {
            "data": {c: [row[c] for row in rows] for c in self.data.columns},
            "index": list(range(len(rows))),
        }

    def to_records(self) -> list[dict]:  # TODO: to be moved to small data
        return [row.asDict(recursive=True) for row in self.data.collect()]

    def loc(self, items: Iterable) -> Iterable:  # TODO: to be moved to small data
        items = Adapter.to_list(items)
        return self.filter(items=items, axis=0)

    def iloc(self, items: Iterable) -> Iterable:  # TODO: to be moved to small data
        items = Adapter.to_list(items)
        return self.filter(items=items, axis=0)


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    def __init__(
        self,
        data: spark.DataFrame | pd.DataFrame | dict | str | None = None,
        session: SparkSession = None,
    ):
        super().__init__(data, session)

    def __deepcopy__(self, memo):
        return SparkDataset(self.data.select("*"))

    @staticmethod
    def _convert_agg_result(result: spark.DataFrame):
        if len(result.columns) == 1 and result.count() == 1:
            return result.limit(1).take(1)[0][result.columns[0]]
        return result

    def apply(
        self,
        func: Callable,
        column_name: str | list[str] | None = None,
        axis: int = 0,
        return_type: T.DataType | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        result_type: Literal["reduce", "expand", "broadcast", None] = None,
    ) -> spark.DataFrame:
        kwargs = kwargs or {}
        df = self.data
        try:
            is_native = isinstance(func(F.col("dummy")), F.Column)
        except Exception:
            is_native = False

        if axis == 0 and is_native:
            cols = Adapter.to_list(column_name)
            if column_name is None:
                new_df = df.select([func(F.col(c)).alias(c) for c in df.columns])
            else:
                new_cols = {c: func(F.col(c)) for c in cols}
                df = df.withColumns(new_cols)
                new_df = df
            return new_df

        if return_type is not None:
            udf_func = F.udf(
                lambda row: func(row.asDict(), *args, **kwargs), return_type
            )
            struct_cols = F.struct(*[F.col(c) for c in (column_name or df.columns)])
            col_expr = udf_func(struct_cols)
            if result_type == "expand" and isinstance(return_type, T.StructType):
                temp_df = df.withColumn("__tmp", col_expr)
                for f in return_type.fields:
                    temp_df = temp_df.withColumn(f.name, F.col("__tmp." + f.name))
                new_df = temp_df.drop("__tmp")
            elif result_type == "reduce":
                new_df = df.withColumn("result", col_expr)
            elif result_type == "broadcast":
                new_df = df.withColumn("result", col_expr)
                if isinstance(return_type, T.StructType):
                    temp_df = new_df.withColumn("__tmp", F.col("result"))
                    for f in return_type.fields:
                        new_df = new_df.withColumn(f.name, F.col("__tmp." + f.name))
                    new_df = new_df.drop("__tmp")
                else:
                    for col in df.columns:
                        new_df = new_df.withColumn(col, F.col("result"))
                    new_df = new_df.drop("result")
            elif result_type is None:
                if isinstance(return_type, T.StructType):
                    temp_df = df.withColumn("__tmp", col_expr)
                    for f in return_type.fields:
                        new_df = temp_df.withColumn(f.name, F.col("__tmp." + f.name))
                    new_df = new_df.drop("__tmp")
                else:
                    new_df = df.withColumn("result", col_expr)
        else:
            self.data.cache()
            df = self.data
            sample_row = df.limit(1).toPandas().iloc[0].to_dict()
            try:
                sample_res = func(sample_row, *args, **kwargs)
            except Exception as e:
                raise ValueError(
                    "Failed to identify return_type by the first line"
                ) from e

            return_type = d.types(sample_res)
            udf_func = F.udf(
                lambda row: func(row.asDict(), *args, **kwargs), return_type
            )
            struct_cols = F.struct(*[F.col(c) for c in (column_name or df.columns)])
            col_expr = udf_func(struct_cols)
            if result_type == "expand" and isinstance(return_type, T.StructType):
                temp_df = df.withColumn("__tmp", col_expr)
                for f in return_type.fields:
                    temp_df = temp_df.withColumn(f.name, F.col("__tmp." + f.name))
                new_df = temp_df.drop("__tmp")
            elif result_type == "reduce":
                new_df = df.withColumn("result", col_expr)
            elif result_type == "broadcast":
                new_df = df.withColumn("result", col_expr)
                if isinstance(return_type, T.StructType):
                    temp_df = new_df.withColumn("__tmp", F.col("result"))
                    for f in return_type.fields:
                        new_df = new_df.withColumn(f.name, F.col("__tmp." + f.name))
                    new_df = new_df.drop("__tmp")
                else:
                    for col in df.columns:
                        new_df = new_df.withColumn(col, F.col("result"))
                    new_df = new_df.drop("result")
            elif result_type is None:
                if isinstance(return_type, T.StructType):
                    temp_df = df.withColumn("__tmp", col_expr)
                    for f in return_type.fields:
                        new_df = temp_df.withColumn(f.name, F.col("__tmp." + f.name))
                    new_df = new_df.drop("__tmp")
                else:
                    new_df = df.withColumn("result", col_expr)
        return new_df

    def map(
        self,
        func: Callable | dict,
        *,
        return_type: T.DataType | None = None,
        na_action: None | str = None,
        **kwargs,
    ) -> spark.DataFrame:
        df = self.data
        cols = df.columns

        if isinstance(func, dict):
            actual_func = func.get
            use_fast_path = False
        else:
            actual_func = func
            use_fast_path = True

        if return_type is not None:
            udf_func = F.udf(actual_func, return_type)
            if na_action == "ignore":
                new_cols = [
                    F.when(F.col(c).isNull(), F.col(c))
                    .otherwise(udf_func(F.col(c)))
                    .alias(c)
                    for c in cols
                ]
            else:
                new_cols = [udf_func(F.col(c)).alias(c) for c in cols]
        else:
            self.data.cache()
            df = self.data
            sample_row = df.limit(1).toPandas().iloc[0].to_dict()
            col_types = {}
            for c in cols:
                try:
                    sample_val = actual_func(sample_row[c], **kwargs)
                except Exception as e:
                    raise ValueError(
                        f"Failed to infer return_type for column '{c}'. "
                        f"Specify it explicitly: return_type=T.StringType()."
                    ) from e
                col_types[c] = d.types(sample_val)

            if na_action == "ignore":
                new_cols = [
                    F.when(F.col(c).isNull(), F.col(c))
                    .otherwise(F.udf(actual_func, col_types[c])(F.col(c)))
                    .alias(c)
                    for c in cols
                ]
            else:
                new_cols = [
                    F.udf(actual_func, col_types[c])(F.col(c)).alias(c) for c in cols
                ]

        new_df = df.select(new_cols)
        return new_df

    def is_empty(self) -> bool:
        return self.data.isEmpty()

    def unique(self) -> SparkDF:
        if not self.data.columns:
            return self.data.select([])
        exprs = [F.collect_set(c).alias(c) for c in self.data.columns]
        return self.data.agg(*exprs)

    def nunique(self, dropna: bool = True) -> SparkDF:
        if not self.data.columns:
            return self.data.select([])
        if dropna:
            exprs = [F.countDistinct(c).alias(c) for c in self.data.columns]
        else:
            exprs = []
            for c in self.data.columns:
                has_null = (F.count("*") - F.count(c) > 0).cast("int")
                exprs.append((F.countDistinct(c) + has_null).alias(c))
        return self.data.agg(*exprs)

    def groupby(self, by: str | Iterable[str], **kwargs) -> list[tuple]:
        if isinstance(by, str):
            by = [by]
        else:
            by = list(by)

        self.data.cache()
        keys_rows = self.data.select(*by).distinct().orderBy(*by).collect()
        result = []
        for row in keys_rows:
            conditions = []
            key_values = []
            for col_name in by:
                val = row[col_name]
                key_values.append(val)
                if val is None:
                    conditions.append(F.col(col_name).isNull())
                else:
                    conditions.append(F.col(col_name) == val)
            final_cond = reduce(lambda x, y: x & y, conditions)
            key = key_values[0] if len(key_values) == 1 else tuple(key_values)
            group_df = self.data.filter(final_cond)
            result.append((key, group_df))
        return result

    def agg(self, func: str | list, **kwargs) -> spark.DataFrame | float:
        df = self.data
        funcs = [func] if isinstance(func, str) else func
        numeric_only = kwargs.get("numeric_only", False)
        ddof = kwargs.get("ddof", 1)
        cols = [
            f.name
            for f in df.schema.fields
            if not numeric_only or isinstance(f.dataType, NumericType)
        ]
        if not cols:
            empty = df.sparkSession.createDataFrame([Row()], StructType([]))
            return self._convert_agg_result(empty)

        special_map = {
            "var": lambda c: F.var_samp(c) if ddof == 1 else F.var_pop(c),
            "std": lambda c: F.stddev_samp(c) if ddof == 1 else F.stddev_pop(c),
        }
        exprs = []
        multi = len(funcs) > 1
        for f_name in funcs:
            spark_fn = None

            if f_name in special_map:
                spark_fn = special_map[f_name]
            else:
                try:
                    spark_fn = getattr(F, f_name)
                except AttributeError:
                    raise ValueError(f"Unsupported agg function: {f_name}")
            for c in cols:
                alias = f"{c}_{f_name}" if multi else c
                if f_name in special_map:
                    col_expr = spark_fn(c).alias(alias)
                else:
                    col_expr = spark_fn(c).alias(alias)
                exprs.append(col_expr)
        df_res = df.agg(*exprs)
        return self._convert_agg_result(df_res)

    def max(self, numeric_only: bool = False):
        return self.agg("max", numeric_only=numeric_only)

    def idxmax(self):
        df = self.data
        if df is None or not df.columns:
            return df

        target_idx = "id"
        df_keyed = df.withColumn(target_idx, F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1)
        numeric_cols = [
            f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)
        ]
        if not numeric_cols:
            return df.select([])

        agg_exprs = [
            F.min(F.struct(-F.col(c), F.col(target_idx).alias("idx"))).alias(c)
            for c in numeric_cols
        ]
        df_agg = df_keyed.agg(*agg_exprs)
        final_exprs = [F.col(c)["idx"].alias(c) for c in numeric_cols]
        return df_agg.select(*final_exprs)

    def min(self, numeric_only: bool = False):
        return self.agg("min", numeric_only=numeric_only)

    def count(self):
        return self.agg("count")

    def sum(self, numeric_only: bool = False):
        return self.agg("sum", numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False):
        return self.agg("mean", numeric_only=numeric_only)

    def mode(self, numeric_only: bool = False, dropna: bool = True) -> SparkDF:
        df = self.data
        if numeric_only:
            cols = [
                f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)
            ]
        else:
            cols = df.columns
        if not cols:
            return df.select([])

        dfs_to_union = []
        for c in cols:
            col_df = df.select(c)
            if dropna:
                col_df = col_df.filter(F.col(c).isNotNull())
            counts_df = col_df.groupBy(c).count()
            max_freq_df = counts_df.agg(F.max("count").alias("max_freq"))
            modes = (
                counts_df.crossJoin(max_freq_df)
                .filter(F.col("count") == F.col("max_freq"))
                .select(c)
            )
            w_order = Window.orderBy(c)
            modes_with_id = modes.withColumn("row_id", F.row_number().over(w_order) - 1)
            dfs_to_union.append(modes_with_id)
        if not dfs_to_union:
            return df.select([])
        final_df = reduce(
            lambda x, y: x.unionByName(y, allowMissingColumns=True), dfs_to_union
        )
        collapse_exprs = [F.first(c, ignorenulls=True).alias(c) for c in cols]
        return (
            final_df.groupBy("row_id")
            .agg(*collapse_exprs)
            .orderBy("row_id")
            .drop("row_id")
        )

    def var(self, numeric_only: bool = False, ddof: int = 1):
        return self.agg("var", numeric_only=numeric_only, ddof=ddof)

    def log(self) -> SparkDF:
        df = self.data
        numeric_cols = [
            f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)
        ]
        exprs = [F.log(F.col(c)).alias(c) for c in numeric_cols]
        if not exprs:
            return df.select([])
        return df.select(*exprs)

    def std(self, numeric_only: bool = False, ddof: int = 1):
        return self.agg("std", numeric_only=numeric_only, ddof=ddof)

    def cov(self, small_format=False):
        col_list = self.data.columns
        paired_cov = self.data.select(
            *[
                F.covar_samp(col_list[i], col_list[j]).alias(f"{i}_{j}")
                for i in range(0, len(col_list))
                for j in range(i, len(col_list))
            ]
        ).toPandas()

        result = pd.DataFrame(
            [
                paired_cov.loc[0, f"{i}_{j}" if i <= j else f"{j}_{i}"]
                for i in range(0, len(col_list))
                for j in range(0, len(col_list))
            ],
            index=col_list,
            columns=col_list,
        )

        if small_format:
            return result
        else:
            return self.data.sparkSession.createDataFrame(result)

    def quantile(self, q: float = 0.5) -> float:
        result = self.data.approxQuantile(self.data.columns[0], q=[q], accuracy=1e-6)[0]
        if isinstance(q, list) and len(q) > 1:
            return result
        self.agg(func="quantile", q=q)

    def coefficient_of_variation(
        self, small_format=False
    ) -> spark.DataFrame | float:
        result = self.data.select(
            F.var_samp(col).alias(col) for col in self.data.columns
        ).toPandas()
        result.index = ["cv"]
        if result.shape[0] == 1 and result.shape[1] == 1:
            return float(result.loc[result.index[0], result.columns[0]])
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)
        if small_format:
            return result
        else:
            return self.data.sparkSession.createDataFrame(result)

    def sort_index(
        self, ascending: bool = True, **kwargs
    ) -> spark.DataFrame:  # Иван # TODO: to be moved to small data
        return self.data.orderBy(F.monotonically_increasing_id(), ascending=ascending)

    def get_numeric_columns(self) -> list[str]:
        return [
            field.name
            for field in self.data.schema.fields
            if isinstance(field.dataType, NumericType)
        ]

    def corr(
        self,
        numeric_only: bool = False,
        small_format: bool = False,
    ) -> spark.DataFrame | float:

        if numeric_only:
            col_list = self.get_numeric_columns()
        else:
            col_list = self.data.columns

        paired_corr = self.data.select(
            *[
                F.corr(col_list[i], col_list[j]).alias(f"{i}_{j}")
                for i in range(0, len(col_list))
                for j in range(i, len(col_list))
            ]
        ).toPandas()

        result = pd.DataFrame(
            [
                paired_corr.loc[0, f"{i}_{j}" if i <= j else f"{j}_{i}"]
                for i in range(0, len(col_list))
                for j in range(0, len(col_list))
            ],
            index=col_list,
            columns=col_list,
        )

        if small_format:
            return result
        else:
            return self.data.sparkSession.createDataFrame(result)

    def isna(self) -> spark.DataFrame:
        return self.data.select(
            *[(F.isnan(col) | F.isnull(col)).alias(col) for col in self.data]
        )

    def sort_values(
        self, by: str | list[str], ascending: bool = True, **kwargs
    ) -> spark.DataFrame:
        by = Adapter.to_list(by)
        return self.data.orderBy(*by, ascending=ascending)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> spark.DataFrame:

        result = self.data
        if dropna:
            result = result.dropna(how="any")

        result = result.select(
            F.count("*").alias("_total_count"),
            *[F.countDistinct(col).alias(col) for col in self.data.columns],
        )

        if normalize:
            result = result.select(
                *[(F.col(col) / F.col("_total_count")).alias(col) for col in self.data]
            )
        else:
            result = result.drop("_total_count")

        if sort:
            raise NotImplementedError

        return result

    def fillna(
        self,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        **kwargs,
    ) -> spark.DataFrame:
        if values is not None and method is not None:
            raise ValueError("Cannot specify both values and method")
        if values is not None:
            return self.data.fillna(values)
        if method is None:
            return self.data

        window = Window.orderBy(F.monotonically_increasing_id())
        if method == "ffill":
            window = window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            expr = lambda c: F.last(F.col(c), ignorenulls=True).over(window)
        elif method == "bfill":
            window = window.rowsBetween(Window.currentRow, Window.unboundedFollowing)
            expr = lambda c: F.first(F.col(c), ignorenulls=True).over(window)
        else:
            raise ValueError(f"Wrong fill method: {method}")

        return self.data.select(*[expr(c).alias(c) for c in self.data.columns])

    def na_counts(self) -> spark.DataFrame | int:
        na_counts = self.data.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in self.data.columns
            ]
        )
        if len(self.data.columns) > 1:
            return na_counts
        else:
            return int(na_counts.collect()[0][0])

    def dot(
        self,
        other: SparkDataset | np.ndarray,
        broadcast_threshold_mb: float = 100.0,
        auto_broadcast: bool = True,
        result_col_prefix: str = "",
    ) -> spark.DataFrame:
        df1 = self.astype({col: float for col in self.columns})

        if isinstance(other, np.ndarray):
            return self._multiply_with_broadcast(
                df1, other, df1.columns, df1.columns, result_col_prefix
            )

        df2 = other.astype({col: float for col in other.columns})

        # Get column lists
        n_cols_df1 = len(df1.columns)
        n_cols_df2 = len(df2.columns)

        # Validate dimensions
        df2_row_count = df2.count()
        if df2_row_count != n_cols_df1:
            raise ValueError(
                f"Matrix dimension mismatch: df1 has {n_cols_df1} columns "
                f"but df2 has {df2_row_count} rows. For matrix multiplication, "
                f"these must be equal."
            )

        # Decide whether to broadcast
        should_broadcast = False
        if auto_broadcast:
            # Estimate df2 size (rough approximation)
            estimated_size_mb = (df2_row_count * n_cols_df2 * 8) / (
                1024 * 1024
            )  # 8 bytes per double
            should_broadcast = estimated_size_mb <= broadcast_threshold_mb
            # print(f"df2 estimated size: {estimated_size_mb:.2f} MB - Broadcasting: {should_broadcast}")

        if should_broadcast:
            result = self._multiply_with_broadcast(
                df1, df2, df1.columns, df2.columns, result_col_prefix
            )
        else:
            result = self._multiply_distributed(
                df1, df2, df1.columns, df2.columns, result_col_prefix
            )

        if isinstance(df2, spark.DataFrame):
            result = result.drop(*list(set(df1.columns) - set(df2.columns)))

        return result

    @staticmethod
    def _multiply_with_broadcast(
        df1: spark.DataFrame,
        df2: spark.DataFrame | np.ndarray,
        df1_cols: list,
        df2_cols: list,
        result_col_prefix: str,
    ) -> spark.DataFrame:
        """
        Multiply using broadcast join - efficient for small df2.
        """

        # Collect df2 as a numpy matrix (it's small enough)
        if isinstance(df2, spark.DataFrame):
            df2_data = df2.select(*df2_cols).collect()
            matrix = np.array([[row[col] for col in df2_cols] for row in df2_data])
        elif isinstance(df2, np.ndarray):
            matrix = df2
        else:
            raise ValueError(
                "The other matrix should be either Dataset or numpy array."
            )

        # Create UDF for matrix multiplication
        @F.udf(ArrayType(DoubleType()))
        def matmul_udf(*row_values):
            row_array = np.array(row_values)
            result = np.dot(row_array, matrix)
            return result.tolist()

        # Apply multiplication
        result_array_col = matmul_udf(*[F.col(c) for c in df1_cols])

        # Expand array into separate columns
        result_df = df1.withColumn("_result_array", result_array_col)

        for i, col_name in enumerate(df2_cols):
            result_df = result_df.withColumn(
                f"{result_col_prefix}{col_name}", result_df["_result_array"][i]
            )

        return result_df.drop("_result_array")

    @staticmethod
    def _multiply_distributed(
        df1: spark.DataFrame,
        df2: spark.DataFrame,
        df1_cols: list,
        df2_cols: list,
        result_col_prefix: str,
    ) -> spark.DataFrame:
        """
        Multiply using distributed joins - for large df2.
        """

        # Add row indices to df1
        df1_indexed = df1.withColumn("_row_id", F.monotonically_increasing_id())

        # Explode df1 into (row_id, col_idx, value) format
        df1_exploded = df1_indexed.select(
            "_row_id",
            F.explode(
                F.array(
                    [
                        F.struct(F.lit(i).alias("_col_idx"), F.col(col).alias("_value"))
                        for i, col in enumerate(df1_cols)
                    ]
                )
            ).alias("_col_struct"),
        ).select(
            "_row_id",
            F.col("_col_struct._col_idx").alias("_col_idx"),
            F.col("_col_struct._value").alias("_value"),
        )

        # Add row index to df2 and explode into (row_idx, col_name, value) format
        df2_indexed = df2.withColumn("_row_idx", F.monotonically_increasing_id())

        df2_exploded = df2_indexed.select(
            "_row_idx",
            F.explode(
                F.array(
                    [
                        F.struct(
                            F.lit(col).alias("_result_col"), F.col(col).alias("_value")
                        )
                        for col in df2_cols
                    ]
                )
            ).alias("_col_struct"),
        ).select(
            F.col("_row_idx").alias("_col_idx"),  # This matches df1's column index
            F.col("_col_struct._result_col").alias("_result_col"),
            F.col("_col_struct._value").alias("_value"),
        )

        # Join and multiply
        joined = df1_exploded.join(df2_exploded, on="_col_idx", how="inner")

        # Multiply values and aggregate
        multiplied = joined.withColumn(
            "_product", F.col("_value") * F.col(df2_exploded["_value"])
        )

        # Group by row_id and result_col, sum products
        aggregated = multiplied.groupBy("_row_id", "_result_col").agg(
            F.sum("_product").alias("_sum")
        )

        # Pivot to get final result
        result = aggregated.groupBy("_row_id").pivot("_result_col").agg(F.first("_sum"))

        # Rename columns with prefix
        for col in df2_cols:
            if col in result.columns:
                result = result.withColumnRenamed(col, f"{result_col_prefix}{col}")

        return result.drop("_row_id").orderBy("_row_id")

    def dropna(  # Эрик
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ) -> spark.DataFrame:
        subset = Adapter.to_list(subset) if subset is not None else None
        if axis in (0, "index", "rows"):
            return self.data.na.drop(how=how, subset=subset)
        if axis in (1, "columns"):
            counts = self.data.select(
                *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in self.columns]
            ).collect()[0].asDict()
            if how == "any":
                keep_cols = [c for c in self.columns if counts[c] == 0]
            else:
                keep_cols = [c for c in self.columns if counts[c] < self.data.count()]
            return self.data.select(*keep_cols)
        raise ValueError("Invalid axis value")

    def transpose(
        self, names: Sequence[str] | None = None
    ) -> spark.DataFrame:  # TODO: to be moved to small data
        pdf = self.data.toPandas().transpose()
        if names is not None:
            pdf.columns = names
        return self.session.createDataFrame(pdf.reset_index(drop=True))

    def sample(  # Эрик
        self,
        frac: float | None = None,
        n: int | None = None,
        random_state: int | None = None,
    ) -> spark.DataFrame:
        if frac is not None and n is not None:
            raise ValueError("Only one of frac or n should be specified")
        if frac is None and n is None:
            raise ValueError("Either frac or n should be specified")
        if frac is not None:
            return self.data.sample(withReplacement=False, fraction=frac, seed=random_state)

        total = self.data.count()
        if total == 0:
            return self.data
        fraction = min(float(n) / float(total), 1.0)
        sampled = self.data.sample(withReplacement=False, fraction=fraction, seed=random_state)
        return sampled.limit(n)

    def select_dtypes(
        self,
        include: str | None = None,
        exclude: str | None = None,
    ) -> spark.DataFrame:
        if include is not None:
            include = Adapter.to_list(include)
            include = [str(v.__name__) if isinstance(v, type) else v for v in include]
        if exclude is not None:
            exclude = Adapter.to_list(exclude)
            exclude = [str(v.__name__) if isinstance(v, type) else v for v in exclude]

        dtypes = self.get_column_type()
        if include is not None:
            dtypes = {k: v for k, v in dtypes.items() if str(v.__name__) in include}
        elif exclude is not None:
            dtypes = {k: v for k, v in dtypes.items() if str(v.__name__) not in exclude}

        return self.data.select([F.col(c).alias(c) for c in dtypes.keys()])

    def limit(self, num: int | None = None) -> Any:
        if not num:
            return self.data
        else:
            return self.data.limit(num)

    def isin(self, values: Iterable) -> spark.DataFrame:
        values = Adapter.to_list(values)
        col_types = self.get_column_type()
        return self.data.select(
            [
                (
                    F.col(c).isin(values).alias(c)
                    if isinstance(values[0], col_types[c])
                    else F.lit(False).alias(c)
                )
                for c in self.data.columns
            ]
        )

    def merge(  # Эрик
        self,
        right: SparkDataset,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool | None = None, 
        right_index: bool | None = None, 
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> spark.DataFrame:
        left_df = self.data
        right_df = right.data

        if on is not None:
            left_on = on
            right_on = on
        if left_index and right_index:
            left_df = left_df.withColumn("__index__", F.monotonically_increasing_id())
            right_df = right_df.withColumn("__index__", F.monotonically_increasing_id())
            left_on = "__index__"
            right_on = "__index__"

        if left_on is None or right_on is None:
            raise MergeOnError(on or left_on or right_on)

        joined = left_df.join(right_df, left_df[left_on] == right_df[right_on], how=how)

        right_renames = {}
        for col in right_df.columns:
            if col == right_on:
                continue
            if col in left_df.columns:
                right_renames[col] = f"{col}{suffixes[1]}"

        result = joined
        for old, new in right_renames.items():
            result = result.withColumnRenamed(old, new)

        if left_on == right_on:
            duplicate_col = f"{right_on}{suffixes[1]}"
            if duplicate_col in result.columns:
                result = result.drop(duplicate_col)

        return result.drop("__index__") if "__index__" in result.columns else result

    def drop(self, labels: Any = "", axis: int = 1) -> spark.DataFrame:
        labels = Adapter.to_list(labels)
        if axis == 1:
            return self.data.drop(*labels)
        elif axis == 0:
            df_with_rownum = self.data.withColumn(
                "__row_num",
                F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
                - 1,
            )
            return df_with_rownum.filter(~F.col("__row_num").isin(labels)).drop(
                "__row_num"
            )
        else:
            raise ValueError("Invalid axis value")

    def filter(
        self,
        items: list | None = None,
        regex: str | None = None,
        column: str | None = None,
        axis: int = 0,
    ) -> spark.DataFrame:
        if axis == 1:
            if items is None and regex is not None:
                items = [col for col in self.data.columns if re.search(regex, col)]
            return self.data.select(items)
        elif axis == 0:
            if items is None and column is not None:
                return self.data.filter(column)
            df_with_rownum = self.data.withColumn(
                "__row_num",
                F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
                - 1,
            )
            return df_with_rownum.filter(F.col("__row_num").isin(items)).drop(
                "__row_num"
            )
        else:
            raise ValueError("Invalid axis value")

    def rename(self, columns: dict[str, str]) -> spark.DataFrame:  # Эрик
        df = self.data
        for old_name, new_name in columns.items():
            df = df.withColumnRenamed(old_name, new_name)
        return df

    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> spark.DataFrame:
        if isinstance(to_replace, dict):
            mgk_mapping_dict = to_replace
        elif isinstance(to_replace, list) and isinstance(value, list):
            mgk_mapping_dict = dict(zip(to_replace, value))
        else:
            mgk_mapping_dict = {to_replace: value}

        mgk_mapping_expr = F.create_map(
            [F.lit(element) for element in chain(*mgk_mapping_dict.items())]
        )

        return self.data.select(
            [
                mgk_mapping_expr[F.col(col_name)].alias(col_name)
                for col_name in self.data.columns
            ]
        )

    def reindex(
        self, labels: str = "", fill_value: str | None = None
    ) -> spark.DataFrame:
        labels = Adapter.to_list(labels)
        existing = self.data.columns
        selected = []
        for c in labels:
            if c in existing:
                selected.append(F.col(c).alias(c))
            else:
                selected.append(F.lit(fill_value).alias(c))
        return self.data.select(*selected)

    def list_to_columns(self, column: str) -> spark.DataFrame:
        n_cols = len(self.data.select(column).head(1).collect()[0][0])

        return self.data.select(
            *[F.col(column)[i].alias(f"{column}_{i}") for i in range(n_cols)]
        )

    def checkpoint(self):
        self.data.checkpoint()