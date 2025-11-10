import os
import sys

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Sequence,
    Sized,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import pyspark.sql as spark
from patsy.util import iterable
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import lit, monotonically_increasing_id

from ...utils import FromDictTypes, MergeOnError, ScalarType, Adapter
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(
        filename: Union[str, Path], session: SparkSession
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
        python_path: Optional[str] = None,
        dynamic_allocation: bool = True,
        mode: Optional[str] = None,
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
        data: Optional[Union[spark.DataFrame, pd.DataFrame, dict, str]] = None,
        session: SparkSession = None,
    ):
        if session is None:
            if isinstance(data, spark.DataFrame):
                self.session = data.spark
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
            self.data = SparkSession.createDataFrame(data)
        elif isinstance(data, str):
            self.data = self._read_file(data, self.session)
        else:
            self.data = SparkSession.emptyDataFrame

    def __getitem__(self, item):

        def get_slice(self, item: slice):
            if isinstance(item, slice):
                start = item.start or 0
                stop = item.stop
                if stop is None:
                    raise ValueError("Slice stop must be defined for Spark navigation")
                step = item.step or 1
                if step <= 0:
                    raise ValueError("Slice step must be positive for Spark navigation")

                indexed = self._with_row_index(self.data)
                condition = (F.col("__row_id") >= start) & (F.col("__row_id") < stop)
                filtered = indexed.filter(condition & ((F.col("__row_id") - start) % step == 0))
                return filtered.drop("__row_id")

        def get_cols(self, item: Union[str, List[str]]):
            columns = Adapter.to_list(item)
            return self.data.select(*columns)

        def get_rows(self, item: Union[int, List[int], slice]):
            if isinstance(item, slice):
                return get_slice(self, item)

            idx = Adapter.to_list(item)
            if any(idx < 0):
                length = self.__len__()
                idx = [length + id if id < 0 else id for id in idx]

            if len(idx) == 1:
                row = self._with_row_index(self.data).filter(F.col("__row_id") == idx).drop("__row_id")
                collected = row.limit(1).collect()
                if not collected:
                    raise IndexError("Spark dataset index out of range")
                return collected[0]
            else:
                df_with_ids = self.withColumn("row_id", monotonically_increasing_id())
                return df_with_ids.filter(df_with_ids.row_id.isin(idx)).drop("row_id")

        sample = item[0] if isinstance(item, List) else item

        if isinstance(sample, str):
            return get_cols(self, item)
        elif isinstance(sample, Union[int, slice]):
            return get_rows(self, item)

        raise KeyError("Unsupported index type for SparkDataset")

    def __len__(self):
        return 0 if self.data is None else self.data.count()

    @staticmethod
    def __magic_determine_other(other) -> Any:
        pass

    # comparison operators:
    def __eq__(self, other) -> Any:
        pass

    def __ne__(self, other) -> Any:
        pass

    def __le__(self, other) -> Any:
        pass

    def __lt__(self, other) -> Any:
        pass

    def __ge__(self, other) -> Any:
        pass

    def __gt__(self, other) -> Any:
        pass

    # Unary operations:
    def __pos__(self) -> Any:
        pass

    def __neg__(self) -> Any:
        pass

    def __abs__(self) -> Any:
        pass

    def __invert__(self) -> Any:
        pass

    def __round__(self, ndigits: int = 0) -> Any:
        pass

    # Binary operations:
    def __add__(self, other) -> Any:
        pass

    def __sub__(self, other) -> Any:
        pass

    def __mul__(self, other) -> Any:
        pass

    def __floordiv__(self, other) -> Any:
        pass

    def __div__(self, other) -> Any:
        pass

    def __truediv__(self, other) -> Any:
        pass

    def __mod__(self, other) -> Any:
        pass

    def __pow__(self, other) -> Any:
        pass

    def __and__(self, other) -> Any:
        pass

    def __or__(self, other) -> Any:
        pass

    # Right arithmetic operators:
    def __radd__(self, other) -> Any:
        pass

    def __rsub__(self, other) -> Any:
        pass

    def __rmul__(self, other) -> Any:
        pass

    def __rfloordiv__(self, other) -> Any:
        pass

    def __rdiv__(self, other) -> Any:
        pass

    def __rtruediv__(self, other) -> Any:
        pass

    def __rmod__(self, other) -> Any:
        pass

    def __rpow__(self, other) -> Any:
        pass

    def __repr__(self):
        pass

    def _repr_html_(self):
        pass

    def get_values(
        self,
        row: Optional[str] = None,
        column: Optional[str] = None,
    ) -> Any:
        if (column is not None) and (row is not None):
            return self[row].data.select(column)
        if column is not None:
            return self[column].data.toPandas.values.tolist()
        if row is not None:
            return self[row].data.toPandas.values.tolist()
        return self

    def iget_values(
        self,
        row: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Any:
        if (column is not None) and (row is not None):
            column = self.columns[column]
            return self[row].data.select(column)
        if column is not None:
            column = self.columns[column]
            return self[column].data.toPandas.values.tolist()
        if row is not None:
            return self[row].data.toPandas.values.tolist()
        return self

    def create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        columns = list(columns or [])
        schema = T.StructType(
            [T.StructField(column, T.StringType(), True) for column in columns]
        )
        empty_rdd = self.session.sparkContext.emptyRDD()
        self.data = self.session.createDataFrame(empty_rdd, schema)
        if index is not None:
            index_rows = [(value,) for value in index]
            index_df = self.session.createDataFrame(index_rows, ["index"])
            self.data = index_df.join(self.data, how="cross") if columns else index_df
        return self

    @property
    def index(self):
        if self.data is None:
            return []
        if "index" in self.data.columns:
            return self.get_values(column="index")
        if "__index__" in self.data.columns:
            return pd.Index(self.data.select("__index__").toPandas()["__index__"])
        return pd.Index(
            [row["__row_id"] for row in self._with_row_index(self.data).select("__row_id").collect()]
        )

    @property
    def columns(self):
        return self.data.columns

    @property
    def shape(self):
        pass

    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        pass

    def get_column_type(
        self, column_name: Union[List[str], str]
    ) -> Optional[Union[Dict[str, type], type]]:
        dtypes = {}
        for k, v in self.data.select(column_name).dtypes:
            if pd.api.types.is_integer_dtype(v):
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
        self, dtype: Dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> spark.DataFrame:
        return self.data.astype(dtype)

    def update_column_type(self, dtype: Dict[str, type]):
        if len(dtype) > 0:
            self.data = self.astype(dtype)
        return self

    def add_column(self, data: Union[spark.DataFrame, List], name: Optional[str] = None, index = None) -> spark.DataFrame:
        def _add_columns_from_dataframe(df: spark.DataFrame, new_df: spark.DataFrame) -> spark.DataFrame:
            if df.count() != new_df.count():
                raise ValueError(
                    f"Row count mismatch: original DF has {df.count()} rows, new DF has {new_df.count()} rows")

            df_with_index = df.withColumn("__join_id", monotonically_increasing_id())
            new_df_with_index = new_df.withColumn("__join_id", monotonically_increasing_id())

            result = df_with_index.join(new_df_with_index, "__join_id", "inner")

            return result.drop("__join_id")

        def _add_columns_from_list(df: spark.DataFrame, data_list: List, column_names: str) -> spark.DataFrame:
            if len(data_list) != df.count():
                raise ValueError(f"Data length {len(data_list)} doesn't match DataFrame row count {df.count()}")

            original_rdd = df.rdd
            zipped_rdd = original_rdd.zip(SparkContext.parallelize(data_list))
            new_df = zipped_rdd.map(lambda x: x[0] + (x[1],)).toDF(df.columns + [column_names])

            return new_df

        if isinstance(data, spark.DataFrame):
            return _add_columns_from_dataframe(self.data, data)
        elif isinstance(data, list):
            return _add_columns_from_list(self.data, data, name)
        else:
            raise ValueError("new_data must be Spark DataFrame, list of values, or list of lists")

    def append(
        self, other, reset_index: bool = False, axis: int = 0
    ) -> spark.DataFrame:
        pass

    def from_dict(
        self, data: FromDictTypes, index: Optional[Union[Iterable, Sized]] = None
    ):
        pass

    def to_dict(self) -> dict[str, Any]:
        pass

    def to_records(self) -> list[dict]:
        pass

    def loc(self, items: Iterable) -> Iterable:
        pass

    def iloc(self, items: Iterable) -> Iterable:
        pass


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    def __init__(
        self,
        data: Optional[Union[spark.DataFrame, pd.DataFrame, dict, str]] = None,
        session: SparkSession = None,
    ):
        super().__init__(data, session)

    @staticmethod
    def _convert_agg_result(result):
        pass

    def apply(self, func: Callable, **kwargs) -> spark.DataFrame:
        pass

    def map(self, func: Callable, **kwargs) -> spark.DataFrame:
        pass

    def is_empty(self) -> bool:
        pass

    def unique(self):
        pass

    def nunique(self, dropna: bool = True):
        pass

    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> list[tuple]:
        pass

    def agg(self, func: Union[str, list], **kwargs) -> Union[spark.DataFrame, float]:
        pass

    def max(self) -> Union[spark.DataFrame, float]:
        pass

    def idxmax(self) -> Union[spark.DataFrame, float]:
        pass

    def min(self) -> Union[spark.DataFrame, float]:
        pass

    def count(self) -> Union[spark.DataFrame, float]:
        pass

    def sum(self) -> Union[spark.DataFrame, float]:
        pass

    def mean(self) -> Union[spark.DataFrame, float]:
        pass

    def mode(
        self, numeric_only: bool = False, dropna: bool = True
    ) -> Union[spark.DataFrame, float]:
        pass

    def var(
        self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False
    ) -> Union[spark.DataFrame, float]:
        pass

    def log(self) -> spark.DataFrame:
        pass

    def std(self, skipna: bool = True, ddof: int = 1) -> Union[spark.DataFrame, float]:
        pass

    def cov(self):
        pass

    def quantile(self, q: float = 0.5) -> spark.DataFrame:
        pass

    def coefficient_of_variation(self) -> Union[spark.DataFrame, float]:
        pass

    def sort_index(self, ascending: bool = True, **kwargs) -> spark.DataFrame:
        pass

    def corr(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        numeric_only: bool = False,
    ) -> Union[spark.DataFrame, float]:
        pass

    def isna(self) -> spark.DataFrame:
        pass

    def sort_values(
        self, by: Union[str, list[str]], ascending: bool = True, **kwargs
    ) -> spark.DataFrame:
        pass

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> spark.DataFrame:
        pass

    def fillna(
        self,
        values: Optional[Union[ScalarType, dict[str, ScalarType]]] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
        **kwargs,
    ) -> spark.DataFrame:
        pass

    def na_counts(self) -> Union[spark.DataFrame, int]:
        pass

    def dot(self, other: Union["SparkDataset", np.ndarray]) -> spark.DataFrame:
        pass

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: Optional[Union[str, Iterable[str]]] = None,
        axis: Union[Literal["index", "rows", "columns"], int] = 0,
    ) -> spark.DataFrame:
        pass

    def transpose(self, names: Optional[Sequence[str]] = None) -> spark.DataFrame:
        pass

    def sample(
        self,
        frac: Optional[float] = None,
        n: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> spark.DataFrame:
        pass

    def select_dtypes(
        self,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> spark.DataFrame:
        pass

    def isin(self, values: Iterable) -> Iterable[bool]:
        pass

    def merge(
        self,
        right: "SparkDataset",
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        left_index: Optional[bool] = None,
        right_index: Optional[bool] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> spark.DataFrame:
        pass

    def drop(self, labels: str = "", axis: int = 1) -> spark.DataFrame:
        pass

    def filter(
        self,
        items: Optional[list] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: int = 0,
    ) -> spark.DataFrame:
        pass

    def rename(self, columns: dict[str, str]) -> spark.DataFrame:
        pass

    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> spark.DataFrame:
        pass

    def reindex(
        self, labels: str = "", fill_value: Optional[str] = None
    ) -> spark.DataFrame:
        pass

    def list_to_columns(self, column: str) -> spark.DataFrame:
        pass
