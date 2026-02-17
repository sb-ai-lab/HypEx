from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Sized, Union

import numpy as np
import pandas as pd
import pyspark.sql as spark
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as F, types as T

from ...utils import FromDictTypes, MergeOnError, ScalarType
from .abstract import DatasetBackendCalc, DatasetBackendNavigation
from .pandas_backend import PandasDataset


class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(filename: Union[str, Path], session: SparkSession) -> spark.DataFrame:
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return (
                session.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(str(filename))
            )
        if file_extension == ".parquet":
            return session.read.parquet(str(filename))
        try:
            return session.read.table(str(filename))
        except Exception as exc:
            raise ValueError("Unsupported file extension {}".format(file_extension)) from exc

    @staticmethod
    def _get_spark_session(
        app_name: str = "HypEx",
        python_path: Optional[str] = None,
        dynamic_allocation: bool = True,
        mode: Optional[str] = None,
    ) -> SparkSession:
        python_path = python_path or sys.executable
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
                .set("spark.executor.cores", "8")
                .set("spark.executor.memory", "8g")
                .set("spark.executor.memoryOverhead", "8g")
                .set("spark.driver.cores", "12")
                .set("spark.driver.memory", "16g")
                .set("spark.driver.maxResultSize", "32g")
                .set("spark.shuffle.service.enabled", "true")
                .set("spark.dynamicAllocation.enabled", str(dynamic_allocation).lower())
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
        session: Optional[SparkSession] = None,
    ):
        self.session = session or self._get_spark_session()

        if isinstance(data, dict):
            if "index" in data:
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
            self.data = self.session.createDataFrame([], T.StructType([]))

    def _to_pandas(self) -> pd.DataFrame:
        return self.data.toPandas()

    def _from_pandas(self, data: pd.DataFrame) -> spark.DataFrame:
        if data.empty and len(data.columns) == 0:
            return self.session.createDataFrame([], T.StructType([]))
        return self.session.createDataFrame(data)

    def __getitem__(self, item):
        if isinstance(item, (str, list)):
            return self.data.select(item)
        if isinstance(item, (slice, int)):
            return self._from_pandas(self._to_pandas().iloc[item])
        if isinstance(item, spark.DataFrame):
            return self.data.join(item)
        raise KeyError("No such column or row")

    def __len__(self):
        return self.data.count()

    @staticmethod
    def __magic_determine_other(other) -> Any:
        if isinstance(other, SparkDataset):
            return other._to_pandas()
        return other

    def _binary_op(self, other, op: Callable[[pd.DataFrame, Any], Any]):
        left = self._to_pandas()
        right = self.__magic_determine_other(other)
        result = op(left, right)
        if isinstance(result, pd.Series):
            result = result.to_frame()
        return self._from_pandas(result)

    def __eq__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l == r)

    def __ne__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l != r)

    def __le__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l <= r)

    def __lt__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l < r)

    def __ge__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l >= r)

    def __gt__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l > r)

    def __pos__(self) -> Any:
        return self._from_pandas(+self._to_pandas())

    def __neg__(self) -> Any:
        return self._from_pandas(-self._to_pandas())

    def __abs__(self) -> Any:
        return self._from_pandas(abs(self._to_pandas()))

    def __invert__(self) -> Any:
        return self._from_pandas(~self._to_pandas())

    def __round__(self, ndigits: int = 0) -> Any:
        return self._from_pandas(round(self._to_pandas(), ndigits))

    def __add__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l + r)

    def __sub__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l - r)

    def __mul__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l * r)

    def __floordiv__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l // r)

    def __div__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l / r)

    def __truediv__(self, other) -> Any:
        return self.__div__(other)

    def __mod__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l % r)

    def __pow__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l**r)

    def __and__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l & r)

    def __or__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: l | r)

    def __radd__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r + l)

    def __rsub__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r - l)

    def __rmul__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r * l)

    def __rfloordiv__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r // l)

    def __rdiv__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r / l)

    def __rtruediv__(self, other) -> Any:
        return self.__rdiv__(other)

    def __rmod__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r % l)

    def __rpow__(self, other) -> Any:
        return self._binary_op(other, lambda l, r: r**l)

    def __repr__(self):
        return self.data.__repr__()

    def _repr_html_(self):
        return self._to_pandas()._repr_html_()

    def create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        self.data = self._from_pandas(pd.DataFrame(index=index, columns=columns))
        return self

    @property
    def index(self):
        return self._to_pandas().index

    @property
    def columns(self):
        return self.data.columns

    @property
    def shape(self):
        return (self.data.count(), len(self.columns))

    def _get_column_index(self, column_name: Union[Sequence[str], str]) -> Union[int, Sequence[int]]:
        columns = list(self.columns)
        if isinstance(column_name, str):
            return columns.index(column_name)
        return [columns.index(col) for col in column_name]

    def get_column_type(
        self, column_name: Union[List[str], str]
    ) -> Optional[Union[Dict[str, type], type]]:
        dtypes: Dict[str, type] = {}
        selected_columns = [column_name] if isinstance(column_name, str) else list(column_name)
        spark_dtype_map = {
            "tinyint": int,
            "smallint": int,
            "int": int,
            "bigint": int,
            "float": float,
            "double": float,
            "decimal": float,
            "string": str,
            "boolean": bool,
            "date": str,
            "timestamp": str,
        }
        for k, v in self.data.select(*selected_columns).dtypes:
            base_type = v.split("(")[0]
            if base_type in spark_dtype_map:
                dtypes[k] = spark_dtype_map[base_type]

        if isinstance(column_name, list):
            return dtypes
        return dtypes.get(column_name)

    def astype(
        self, dtype: Dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> spark.DataFrame:
        type_map = {int: "bigint", float: "double", str: "string", bool: "boolean"}
        result = self.data
        for col_name, col_type in dtype.items():
            if col_type not in type_map:
                if errors == "raise":
                    raise ValueError("Unsupported type cast for column {}".format(col_name))
                continue
            result = result.withColumn(col_name, F.col(col_name).cast(type_map[col_type]))
        return result

    def update_column_type(self, dtype: Dict[str, type]):
        if dtype:
            self.data = self.astype(dtype)
        return self

    def add_column(self, data: Union[spark.DataFrame, List], name: Optional[Union[str, List[str]]] = None, index=None):
        if isinstance(name, list) and len(name) == 1:
            name = name[0]
        if isinstance(data, spark.DataFrame):
            if self.data.count() != data.count():
                raise ValueError("Row count mismatch")
            left = self.data.rdd.zipWithIndex().map(lambda x: (x[1],) + tuple(x[0]))
            right = data.rdd.zipWithIndex().map(lambda x: (x[1],) + tuple(x[0]))
            left_schema = ["__idx"] + self.columns
            right_schema = ["__idx"] + [c for c in data.columns if c not in self.columns]
            left_df = self.session.createDataFrame(left, left_schema)
            right_df = self.session.createDataFrame(right, right_schema)
            self.data = left_df.join(right_df, on="__idx", how="inner").drop("__idx")
            return

        if isinstance(data, list):
            if len(data) != self.data.count():
                raise ValueError("Data length doesn't match DataFrame row count")
            col_name = name if isinstance(name, str) else "new_column"
            add_df = self.session.createDataFrame([(v,) for v in data], schema=[col_name])
            self.add_column(add_df)
            return

        raise ValueError("new_data must be Spark DataFrame or list")

    def append(self, other, reset_index: bool = False, axis: int = 0) -> spark.DataFrame:
        if axis != 0:
            all_frames = [self._to_pandas()] + [d._to_pandas() for d in other]
            result = pd.concat(all_frames, axis=axis)
            return self._from_pandas(result.reset_index(drop=True) if reset_index else result)

        result = self.data
        for item in other:
            result = result.unionByName(item.data, allowMissingColumns=True)
        if reset_index:
            return self._from_pandas(result.toPandas().reset_index(drop=True))
        return result

    def from_dict(self, data: FromDictTypes, index: Optional[Union[Iterable, Sized]] = None):
        pdf = pd.DataFrame.from_records(data, columns=list(data.keys()) if isinstance(data, dict) else None)
        if index is not None:
            pdf.index = index
        self.data = self._from_pandas(pdf)
        return self

    def to_dict(self) -> Dict[str, Any]:
        pdf = self._to_pandas()
        return {"data": {column: pdf[column].to_list() for column in pdf.columns}, "index": list(pdf.index)}

    def to_records(self) -> List[dict]:
        return self._to_pandas().to_dict(orient="records")

    def loc(self, items: Iterable) -> Iterable:
        return self._from_pandas(self._to_pandas().loc[items])

    def iloc(self, items: Iterable) -> Iterable:
        return self._from_pandas(self._to_pandas().iloc[items])


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    def __init__(
        self,
        data: Optional[Union[spark.DataFrame, pd.DataFrame, dict, str]] = None,
        session: Optional[SparkSession] = None,
    ):
        super().__init__(data, session)

    @staticmethod
    def _convert_agg_result(result):
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if isinstance(result, pd.DataFrame) and result.shape == (1, 1):
            return float(result.iloc[0, 0])
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def _delegate_pandas(self, method_name: str, *args, **kwargs):
        pandas_backend = PandasDataset(self._to_pandas())
        result = getattr(pandas_backend, method_name)(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            return self._from_pandas(result)
        if isinstance(result, pd.Series):
            return self._from_pandas(result.to_frame())
        return result

    def get_values(self, row: Optional[str] = None, column: Optional[str] = None) -> Any:
        return PandasDataset(self._to_pandas()).get_values(row=row, column=column)

    def iget_values(self, row: Optional[int] = None, column: Optional[int] = None) -> Any:
        return PandasDataset(self._to_pandas()).iget_values(row=row, column=column)

    def apply(self, func: Callable, **kwargs) -> spark.DataFrame:
        return self._delegate_pandas("apply", func, **kwargs)

    def map(self, func: Callable, **kwargs) -> spark.DataFrame:
        return self._delegate_pandas("map", func, **kwargs)

    def is_empty(self) -> bool:
        return self.data.rdd.isEmpty()

    def unique(self):
        return PandasDataset(self._to_pandas()).unique()

    def nunique(self, dropna: bool = True):
        return PandasDataset(self._to_pandas()).nunique(dropna=dropna)

    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> list:
        return PandasDataset(self._to_pandas()).groupby(by=by, **kwargs)

    def agg(self, func: Union[str, list], **kwargs) -> Union[spark.DataFrame, float]:
        result = PandasDataset(self._to_pandas()).agg(func, **kwargs)
        if isinstance(result, pd.DataFrame):
            return self._from_pandas(result)
        return result

    def max(self) -> Union[spark.DataFrame, float]:
        return self.agg(["max"])

    def idxmax(self) -> Union[spark.DataFrame, float]:
        return self.agg(["idxmax"])

    def min(self) -> Union[spark.DataFrame, float]:
        return self.agg(["min"])

    def count(self) -> Union[spark.DataFrame, float]:
        return self.agg(["count"])

    def sum(self) -> Union[spark.DataFrame, float]:
        return self.agg(["sum"])

    def mean(self) -> Union[spark.DataFrame, float]:
        return self.agg(["mean"])

    def mode(self, numeric_only: bool = False, dropna: bool = True) -> Union[spark.DataFrame, float]:
        return self._delegate_pandas("mode", numeric_only=numeric_only, dropna=dropna)

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> Union[spark.DataFrame, float]:
        return self.agg(["var"], skipna=skipna, ddof=ddof, numeric_only=numeric_only)

    def log(self) -> spark.DataFrame:
        return self._delegate_pandas("log")

    def std(self, skipna: bool = True, ddof: int = 1) -> Union[spark.DataFrame, float]:
        return self.agg(["std"], skipna=skipna, ddof=ddof)

    def cov(self):
        return self._delegate_pandas("cov")

    def quantile(self, q: float = 0.5) -> spark.DataFrame:
        return self._delegate_pandas("quantile", q=q)

    def coefficient_of_variation(self) -> Union[spark.DataFrame, float]:
        result = PandasDataset(self._to_pandas()).coefficient_of_variation()
        return self._from_pandas(result) if isinstance(result, pd.DataFrame) else result

    def sort_index(self, ascending: bool = True, **kwargs) -> spark.DataFrame:
        return self._delegate_pandas("sort_index", ascending=ascending, **kwargs)

    def corr(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        numeric_only: bool = False,
    ) -> Union[spark.DataFrame, float]:
        return self._delegate_pandas("corr", method=method, numeric_only=numeric_only)

    def isna(self) -> spark.DataFrame:
        return self._delegate_pandas("isna")

    def sort_values(self, by: Union[str, List[str]], ascending: bool = True, **kwargs) -> spark.DataFrame:
        return self._delegate_pandas("sort_values", by=by, ascending=ascending, **kwargs)

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, dropna: bool = True) -> spark.DataFrame:
        return self._delegate_pandas(
            "value_counts",
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna,
        )

    def fillna(
        self,
        values: Optional[Union[ScalarType, Dict[str, ScalarType]]] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
        **kwargs,
    ) -> spark.DataFrame:
        return self._delegate_pandas("fillna", values=values, method=method, **kwargs)

    def na_counts(self) -> Union[spark.DataFrame, int]:
        result = PandasDataset(self._to_pandas()).na_counts()
        return self._from_pandas(result) if isinstance(result, pd.DataFrame) else result

    def dot(self, other: Union["SparkDataset", np.ndarray]) -> spark.DataFrame:
        if isinstance(other, SparkDataset):
            return self._from_pandas(PandasDataset(self._to_pandas()).dot(PandasDataset(other._to_pandas())))
        return self._from_pandas(PandasDataset(self._to_pandas()).dot(other))

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: Optional[Union[str, Iterable[str]]] = None,
        axis: Union[Literal["index", "rows", "columns"], int] = 0,
    ) -> spark.DataFrame:
        return self._delegate_pandas("dropna", how=how, subset=subset, axis=axis)

    def transpose(self, names: Optional[Sequence[str]] = None) -> spark.DataFrame:
        return self._delegate_pandas("transpose", names=names)

    def sample(
        self,
        frac: Optional[float] = None,
        n: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> spark.DataFrame:
        return self._delegate_pandas("sample", frac=frac, n=n, random_state=random_state)

    def select_dtypes(self, include: Optional[str] = None, exclude: Optional[str] = None) -> spark.DataFrame:
        return self._delegate_pandas("select_dtypes", include=include, exclude=exclude)

    def isin(self, values: Iterable) -> Iterable[bool]:
        return self._delegate_pandas("isin", values)

    def merge(
        self,
        right: "SparkDataset",
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        left_index: Optional[bool] = None,
        right_index: Optional[bool] = None,
        suffixes: tuple = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> spark.DataFrame:
        for on_ in [on, left_on, right_on]:
            if on_ and (
                on_ not in [*self.columns, *right.columns]
                if isinstance(on_, str)
                else any(c not in [*self.columns, *right.columns] for c in on_)
            ):
                raise MergeOnError(on_)

        if not all([on, left_on, right_on]) and all([left_index is None, right_index is None]):
            left_index = True
            right_index = True

        left_pd = self._to_pandas()
        right_pd = right._to_pandas()
        result = left_pd.merge(
            right=right_pd,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            how=how,
        )
        return self._from_pandas(result)

    def drop(self, labels: str = "", axis: int = 1) -> spark.DataFrame:
        return self._delegate_pandas("drop", labels=labels, axis=axis)

    def filter(self, items: Optional[list] = None, like: Optional[str] = None, regex: Optional[str] = None, axis: int = 0) -> spark.DataFrame:
        return self._delegate_pandas("filter", items=items, like=like, regex=regex, axis=axis)

    def rename(self, columns: Dict[str, str]) -> spark.DataFrame:
        return self._delegate_pandas("rename", columns=columns)

    def replace(self, to_replace: Any = None, value: Any = None, regex: bool = False) -> spark.DataFrame:
        return self._delegate_pandas("replace", to_replace=to_replace, value=value, regex=regex)

    def reindex(self, labels: str = "", fill_value: Optional[str] = None) -> spark.DataFrame:
        return self._delegate_pandas("reindex", labels=labels, fill_value=fill_value)

    def list_to_columns(self, column: str) -> spark.DataFrame:
        return self._delegate_pandas("list_to_columns", column)
