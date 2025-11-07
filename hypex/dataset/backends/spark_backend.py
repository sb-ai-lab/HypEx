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
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import lit, monotonically_increasing_id

from pyspark.sql import DataFrame as SparkDF
from pyspark.sql.column import Column


from ...utils import FromDictTypes, MergeOnError, ScalarType
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
        pass

    def __len__(self):
        pass

    @staticmethod
    def __magic_determine_other(other) -> Any:
        if isinstance(other, (int, float, str, bool)):
            return F.lit(other)
        elif isinstance(other, F.Column):
            return other
        else:
            raise TypeError(
                f"Unsupported operand type: '{type(other).__name__}'. "
                )
        
    # comparison operators:
    def __eq__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) == other).alias(c) for c in self.data.columns)

    def __ne__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) != other).alias(c) for c in self.data.columns)

    def __lt__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) < other).alias(c) for c in self.data.columns)

    def __le__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) <= other).alias(c) for c in self.data.columns)

    def __ge__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) >= other).alias(c) for c in self.data.columns)
    
    def __gt__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) > other).alias(c) for c in self.data.columns)

    # Unary operations:
    def __pos__(self) -> spark.DataFrame:
        return self.data.select("*")

    def __neg__(self) -> spark.DataFrame:
        return self.data.select((-F.col(c)).alias(c) for c in self.data.columns)

    def __abs__(self) -> spark.DataFrame:
        return self.data.select(F.abs(F.col(c)).alias(c) for c in self.data.columns)

    def __invert__(self) -> spark.DataFrame:
        return self.data.select((~F.col(c)).alias(c) for c in self.data.columns)

    def __round__(self, ndigits: int = 0) -> spark.DataFrame:
        return self.data.select(F.round(F.col(c), ndigits).alias(c) for c in self.data.columns)

    # Binary operations:
    def __add__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) + other).alias(c) for c in self.data.columns)

    def __sub__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) - other).alias(c) for c in self.data.columns)

    def __mul__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) * other).alias(c) for c in self.data.columns)

    def __floordiv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.floor(F.col(c) / other)).alias(c) for c in self.data.columns)

    def __mod__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) % other).alias(c) for c in self.data.columns)

    def __pow__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select(F.pow(F.col(c), other).alias(c) for c in self.data.columns)

    def __and__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) & other).alias(c) for c in self.data.columns)

    def __or__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.col(c) | other).alias(c) for c in self.data.columns)

    # Right arithmetic operators:
    def __radd__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((other + F.col(c)).alias(c) for c in self.data.columns)

    def __rsub__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((other - F.col(c)).alias(c) for c in self.data.columns)

    def __rmul__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((other * F.col(c)).alias(c) for c in self.data.columns)

    def __rfloordiv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((F.floor(other / F.col(c))).alias(c) for c in self.data.columns)

    def __rdiv__(self, other) -> spark.DataFrame:
        return self.__rtruediv__(other)

    def __rtruediv__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((other / F.col(c)).alias(c) for c in self.data.columns)

    def __rmod__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select((other % F.col(c)).alias(c) for c in self.data.columns)

    def __rpow__(self, other) -> spark.DataFrame:
        other = self.__magic_determine_other(other)
        return self.data.select(F.pow(other, F.col(c)).alias(c) for c in self.data.columns)

    def __repr__(self):
        return self.data.__repr__()

    def _repr_html_(self):
        return self.data._repr_html_()

    def create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        pass

    @property
    def index(self):
        pass

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
            # elif pd.api.types.is_object_dtype(v) and pd.api.types.is_list_like(
            #     self.data[column_name].iloc[0]
            # ):
            #     dtypes[k] = object
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
            zipped_rdd = original_rdd.zip(spark.sparkContext.parallelize(data_list))
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

    def get_values(
        self,
        row: Optional[str] = None,
        column: Optional[str] = None,
    ) -> Any:
        pass

    def iget_values(
        self,
        row: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Any:
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
