from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Sequence, Sized, Union, Optional

from functools import reduce

import numpy as np
import pandas as pd

import pyspark.pandas as ps

import pyspark.sql as spark
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDF

from pyspark.sql.types import StructType

from ...utils import FromDictTypes, MergeOnError, ScalarType
from .abstract import DatasetBackendCalc, DatasetBackendNavigation

class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(filename: str | Path, session: SparkSession) -> ps.DataFrame:
        file_path = Path(filename).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: '{file_path}'")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: '{file_path}'")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied: '{file_path}'")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".csv":
            spark_df = (session
                .read
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("encoding", "UTF-8")
                .load(str(file_path))
            )
        elif suffix == ".parquet":
            spark_df = session.read.parquet(str(file_path))
        elif suffix == ".json":
            spark_df = session.read.json(str(file_path))
        elif suffix == ".orc":
            spark_df = session.read.orc(str(file_path))
        elif suffix == ".xlsx":
            return ps.read_excel(str(file_path))
        else:
            raise ValueError(f"Unsupported file extension: '{suffix}'. "
                             f"Supported: .csv, .parquet, .json, .orc, .xlsx")
        
        return ps.DataFrame(spark_df)

    def __init__(self,
                 data: ps.DataFrame | SparkDF | pd.DataFrame | dict[str, Any] | str | None = None,
                 session: SparkSession | None = None):        
        if session is None:
            raise ValueError(f"Session not set")
        elif isinstance(session, SparkSession):
            self.session = session
        else:
            raise TypeError("Session must be an instance of SparkSession")
        
        ps.set_option('compute.ops_on_diff_frames', True)
        
        if isinstance(data, dict):
            if "index" in data:
                data = pd.DataFrame(data=data["data"], index=data["index"])
            else:
                data = pd.DataFrame(data=data["data"])

        if isinstance(data, ps.DataFrame):
            self.data = data
        elif isinstance(data, SparkDF):
            self.data = ps.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            spark_df = self.session.createDataFrame(data)
            self.data = ps.DataFrame(spark_df)
        elif isinstance(data, (ps.Series, pd.Series)):
            temp_df = pd.DataFrame(data)
            spark_df = self.session.createDataFrame(temp_df.reset_index())
            self.data = ps.DataFrame(spark_df)
        elif isinstance(data, str):
            self.data = self._read_file(data, self.session)
        elif data is None:
            self.data = ps.DataFrame(self.session.createDataFrame([], schema=StructType([])))
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.data.iloc[item]
        if isinstance(item, (str, list)):
            return self.data[item]
        if isinstance(item, ps.DataFrame):
            if len(item.columns) == 1:
                return self.data[item.iloc[:, 0]]
            else:
                return self.data[item]
        if isinstance(item, ps.Series):
            return self.__class__(data=self.data[item], session=self.session)
        raise KeyError("No such column or row")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def __magic_determine_other(other) -> Any:
        if isinstance(other, SparkDataset):
            return other.data
        else:
            return other

    # comparison operators:
    def __eq__(self, other) -> Any:
        return self.data == self.__magic_determine_other(other)

    def __ne__(self, other) -> Any:
        return self.data != self.__magic_determine_other(other)

    def __le__(self, other) -> Any:
        return self.data <= self.__magic_determine_other(other)

    def __lt__(self, other) -> Any:
        return self.data < self.__magic_determine_other(other)

    def __ge__(self, other) -> Any:
        return self.data >= self.__magic_determine_other(other)

    def __gt__(self, other) -> Any:
        return self.data > self.__magic_determine_other(other)

    # Unary operations:
    def __pos__(self) -> Any:
        result = +self.data
        return self.__class__(data=result, session=self.session)


    def __neg__(self) -> Any:
        result = -self.data
        return self.__class__(data=result, session=self.session)


    def __abs__(self) -> Any:
        result = abs(self.data)
        return self.__class__(data=result, session=self.session)


    def __invert__(self) -> Any:
        result = ~self.data
        return self.__class__(data=result, session=self.session)


    def __round__(self, ndigits: int = 0) -> Any:
        result = self.data.round(ndigits)
        return self.__class__(data=result, session=self.session)


    # Binary operations:
    def __add__(self, other) -> Any:
        result = self.data + self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __sub__(self, other) -> Any:
        result = self.data - self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __mul__(self, other) -> Any:
        result = self.data * self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __floordiv__(self, other) -> Any:
        result = self.data // self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __div__(self, other) -> Any:
        result = self.data / self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __truediv__(self, other) -> Any:
        result = self.data / self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __mod__(self, other) -> Any:
        result = self.data % self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __pow__(self, other) -> Any:
        result = self.data ** self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __and__(self, other) -> Any:
        result = self.data & self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    def __or__(self, other) -> Any:
        result = self.data | self.__magic_determine_other(other)
        return self.__class__(data=result, session=self.session)

    # Right arithmetic operators:
    def __radd__(self, other) -> Any:
        result = self.__magic_determine_other(other) + self.data
        return self.__class__(data=result, session=self.session)

    def __rsub__(self, other) -> Any:
        result = self.__magic_determine_other(other) - self.data
        return self.__class__(data=result, session=self.session)

    def __rmul__(self, other) -> Any:
        result = self.__magic_determine_other(other) * self.data
        return self.__class__(data=result, session=self.session)

    def __rfloordiv__(self, other) -> Any:
        result = self.__magic_determine_other(other) // self.data
        return self.__class__(data=result, session=self.session)

    def __rdiv__(self, other) -> Any:
        result = self.__magic_determine_other(other) / self.data
        return self.__class__(data=result, session=self.session)

    def __rtruediv__(self, other) -> Any:
        result = self.__magic_determine_other(other) / self.data
        return self.__class__(data=result, session=self.session)

    def __rmod__(self, other) -> Any:
        result = self.__magic_determine_other(other) % self.data
        return self.__class__(data=result, session=self.session)

    def __rpow__(self, other) -> Any:
        result = self.__magic_determine_other(other) ** self.data
        return self.__class__(data=result, session=self.session)

    def __repr__(self):
        return self.data.__repr__()

    def _repr_html_(self):
        return self.data._repr_html_()

    def get_values(self, row: str | None = None, column: str | None = None) -> Any:
        if (column is not None) and (row is not None):
            return self.data.loc[row, column]
        elif column is not None:
            result = self.data.loc[:, column]
        elif row is not None:
            result = self.data.loc[row, :]
        else:
            result = self.data
        
        if isinstance(result, (ps.DataFrame, ps.Series)):
            return result.to_pandas().values.tolist()
        return result

    def iget_values(self, row: int | None = None, column: int | None = None) -> Any:
        if (column is not None) and (row is not None):
            return self.data.iloc[row, column]
        elif column is not None:
            result = self.data.iloc[:, column]
        elif row is not None:
            result = self.data.iloc[row, :]
        else:
            result = self.data
            
        if isinstance(result, (ps.DataFrame, ps.Series)):
            return result.to_pandas().values.tolist()
        return result

    def create_empty(self, index: Iterable | None = None, columns: Iterable[str] | None = None):
        self.data = ps.DataFrame(index=index, columns=columns)
        return self

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns.tolist()

    @property
    def shape(self):
        return self.data.shape

    def _get_column_index(self, column_name: Sequence[str] | str) -> int | Sequence[int]:
        if isinstance(column_name, str):
            return self.data.columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return self.data.columns.get_indexer(column_name)
        else:
            raise ValueError("Wrong column_name type.")

    def get_column_type(self, column_name: Iterable[str] | str = None) -> dict[str, type] | type | None:
        column_name = self.data.columns if column_name is None else column_name
        pdf = self.data.to_pandas()
        
        if isinstance(column_name, str):
            dtype = pdf[column_name].dtype
            if pd.api.types.is_integer_dtype(dtype):
                return int
            elif pd.api.types.is_float_dtype(dtype):
                return float
            elif pd.api.types.is_bool_dtype(dtype):
                return bool
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                return str
            return type(pdf[column_name].iloc[0]) if len(pdf) > 0 else None
        
        dtypes = {}
        for col in column_name:
            dtype = pdf[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                dtypes[col] = int
            elif pd.api.types.is_float_dtype(dtype):
                dtypes[col] = float
            elif pd.api.types.is_bool_dtype(dtype):
                dtypes[col] = bool
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                dtypes[col] = str
            else:
                dtypes[col] = type(pdf[col].iloc[0]) if len(pdf) > 0 else object
        
        return dtypes

    def astype(self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise") -> SparkDataset:
        result = self.data.astype(dtype=dtype)
        return self.__class__(data=result, session=self.session)

    def update_column_type(self, dtype: Dict[str, type]):
        for column_name, type_name in dtype.items():
            if not self.data[column_name].isna().any():
                self.data = self.astype({column_name: type_name}).data
        return self

    def add_column(self, data: Sequence, name: str | list[str], index: Sequence | None = None):
        if isinstance(name, list) and len(name) == 1:
            name = name[0]
        if isinstance(data, ps.DataFrame):
            data = data.to_pandas().values
        if len(self.data) != len(data):
            if isinstance(data[0], Iterable) and len(data[0]) == 1:
                data = data.squeeze()
            data = ps.Series(data)
        
        if index:
            self.data[name] = data
        else:
            self.data[name] = data
        return self

    def append(self, other, reset_index: bool = False, axis: int = 0) -> SparkDataset:
        new_data = ps.concat([self.data] + [d.data for d in other], axis=axis)
        if reset_index:
            new_data = new_data.reset_index(drop=True)
        return self.__class__(data=new_data, session=self.session)

    def from_dict(self, data: FromDictTypes, index: Iterable | Sized | None = None):
        if isinstance(data, dict):
            self.data = ps.DataFrame().from_records(data, columns=list(data.keys()))
        else:
            self.data = ps.DataFrame().from_records(data)
        if index is not None:
            self.data.index = index
        return self

    def to_dict(self) -> dict[str, Any]:
        pdf = self.data.to_pandas()
        return {
            "data": {
                column: pdf[column].to_list() for column in pdf.columns
            },
            "index": list(pdf.index),
        }

    def to_records(self) -> list[dict]:
        return self.data.to_pandas().to_dict(orient="records")

    def loc(self, items: Iterable) -> SparkDataset:
        data = self.data.loc[items]
        if not isinstance(data, ps.DataFrame):
            data = ps.DataFrame(data)
        return self.__class__(data=data, session=self.session)

    def iloc(self, items: Iterable) -> SparkDataset:
        data = self.data.iloc[items]
        if not isinstance(data, ps.DataFrame):
            data = ps.DataFrame(data)
        return self.__class__(data=data, session=self.session)


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    @staticmethod
    def _convert_agg_result(result):
        if isinstance(result, ps.Series):
            result = result.to_frame()
        if result.shape == (1, 1):
            return float(result.to_spark().collect()[0][0])
        return result if isinstance(result, ps.DataFrame) else ps.DataFrame(result)

    def __init__(self, 
                 data: ps.DataFrame | dict | str | ps.Series | None = None,
                 session: SparkSession | None = None):
        super().__init__(data=data, session=session)

    def get(self, key, default=None) -> Any:
        return self.data.get(key, default)

    def take(self, indices: int | list[int], axis: Literal["index", "columns", "rows"] | int = 0) -> Any:
        return self.data.take(indices=indices, axis=axis)

    def apply(self, func: Callable, **kwargs) -> SparkDataset:
        single_column_name = kwargs.pop("column_name", None)
        result = self.data.apply(func, **kwargs)
        if not isinstance(result, ps.DataFrame):
            result = result.to_frame(name=single_column_name)
        return self.__class__(data=result, session=self.session)

    def map(self, func: Callable, **kwargs) -> SparkDataset:
        result = self.data.apply(lambda col: col.apply(func, **kwargs))
        return self.__class__(data=result, session=self.session)

    def is_empty(self) -> bool:
        return self.data.empty

    def unique(self):
        return {column: self.data[column].unique().to_pandas().tolist() for column in self.data.columns}

    def nunique(self, dropna: bool = True):
        return {column: self.data[column].nunique() for column in self.data.columns}

    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> list[tuple]:
        if isinstance(by, str):
            by = [by]
        else:
            by = list(by)

        keys_pdf = self.data[by].drop_duplicates().sort_values(by).to_pandas()

        result = []
        
        spark_df = self.data._internal.spark_frame
        spark_df.cache()
        
        try:
            for _, row in keys_pdf.iterrows():
                conditions = []
                key_values = []
                
                for col_name in by:
                    val = row[col_name]
                    key_values.append(val)
                    
                    if pd.isna(val):
                        conditions.append(self.data[col_name].isna())
                    else:
                        conditions.append(self.data[col_name] == val)
                
                if conditions:
                    final_cond = reduce(lambda x, y: x & y, conditions)
                    group_df = self.data[final_cond]
                else:
                    group_df = self.data
                
                key = key_values[0] if len(key_values) == 1 else tuple(key_values)
                
                result.append((key, SparkDataset(group_df, self.session)))
                
        finally:
            spark_df.unpersist()

        return result

    def agg(self, func: str | list, **kwargs) -> SparkDataset | float:
        subset = kwargs.pop('subset', None)
        
        func = func if isinstance(func, (list, dict)) else [func]
        
        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            data_to_agg = self.data[subset]
        else:
            data_to_agg = self.data
        
        result = data_to_agg.agg(func, **kwargs)
        converted = self._convert_agg_result(result)
        
        if isinstance(converted, ps.DataFrame):
            return self.__class__(data=converted, session=self.session)
        return converted

    def max(self) -> SparkDataset | float:
        return self.agg(["max"])

    def idxmax(self) -> SparkDataset | float:
        return self.agg(["idxmax"])

    def min(self) -> SparkDataset | float:
        return self.agg(["min"])

    def count(self) -> SparkDataset | float:
        return self.agg(["count"])

    def sum(self) -> SparkDataset | float:
        return self.agg(["sum"])

    def mean(self) -> SparkDataset | float:
        return self.agg(["mean"])

    def mode(
        self, numeric_only: bool = False, dropna: bool = True
    ) -> SparkDataset:
        result = self.data.mode(numeric_only=numeric_only, dropna=dropna)
        return self.__class__(data=result, session=self.session)

    def std(self, skipna: bool = True, ddof: int = 1) -> SparkDataset | float:
        result = self.data.std()
        converted = self._convert_agg_result(result.to_frame() if isinstance(result, ps.Series) else result)
        if isinstance(converted, ps.DataFrame):
            return self.__class__(data=converted, session=self.session)
        return converted

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> SparkDataset | float:
        result = self.data.var()
        converted = self._convert_agg_result(result.to_frame() if isinstance(result, ps.Series) else result)
        if isinstance(converted, ps.DataFrame):
            return self.__class__(data=converted, session=self.session)
        return converted

    def log(self) -> SparkDataset:
        np_data = np.log(self.data.to_numpy())
        result = ps.DataFrame(np_data, columns=self.data.columns)
        return self.__class__(data=result, session=self.session)

    def cov(self) -> SparkDataset:
        result = self.data.cov()
        return self.__class__(data=result, session=self.session)

    def quantile(self, q: float = 0.5) -> SparkDataset | float:
        if isinstance(q, list) and len(q) > 1:
            result = self.data.quantile(q=q)
            return self.__class__(data=result, session=self.session)
        else:
            result = self.data.quantile(q=q)
            converted = self._convert_agg_result(result.to_frame() if isinstance(result, ps.Series) else result)
            if isinstance(converted, ps.DataFrame):
                return self.__class__(data=converted, session=self.session)
            return converted

    def coefficient_of_variation(self) -> SparkDataset | float:
        std_series = self.data.std()
        mean_series = self.data.mean()
        cv_series = std_series / mean_series
        cv_df = cv_series.to_frame().T
        
        if cv_df.shape == (1, 1):
            return float(cv_df.to_spark().collect()[0][0])
        
        old_index_name = list(cv_df.index)[0]
        cv_df = cv_df.rename(index={old_index_name: "cv"})
        
        return self.__class__(data=cv_df, session=self.session)

    def sort_index(self, ascending: bool = True, **kwargs) -> SparkDataset:
        result = self.data.sort_index(ascending=ascending, **kwargs)
        return self.__class__(data=result, session=self.session)

    def get_numeric_columns(self) -> list[str]:
        types = self.get_column_type()
        return [col for col, dtype in types.items() if dtype in [int, float, np.int64, np.float64]]

    def corr(self, numeric_only: bool = False) -> SparkDataset | float:
        result = self.data.corr(method='pearson') 
        
        if isinstance(result, ps.DataFrame):
            return self.__class__(data=result, session=self.session)
        return result

    def isna(self) -> SparkDataset:
        result = self.data.isna()
        return self.__class__(data=result, session=self.session)

    def sort_values(self, by: str | list[str], ascending: bool = True, **kwargs) -> SparkDataset:
        result = self.data.sort_values(by=by, ascending=ascending, **kwargs)
        return self.__class__(data=result, session=self.session)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> SparkDataset:
        col = list(self.data.columns)[0]
        series = self.data[col]
        
        result = series.value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna
        )
        
        result_df = result.to_frame(name="count").reset_index()
        result_df = result_df.rename(columns={col: "value", "index": "value"})
        
        return self.__class__(data=result_df, session=self.session)

    def na_counts(self) -> SparkDataset | int:
        data = self.data.isna().sum().to_frame().T
        
        if data.shape[0] == 1 and data.shape[1] == 1:
            return int(data.to_spark().collect()[0][0])
        
        old_index_name = list(data.index)[0]
        data = data.rename(index={old_index_name: "na_counts"})
        
        return self.__class__(data=data, session=self.session)

    def dot(self, other: 'SparkDataset' | np.ndarray) -> SparkDataset:
        result_df: ps.DataFrame

        if isinstance(other, np.ndarray):
            other_transposed = other.T
            
            index_names = self.columns if other.shape[1] == self.shape[1] else None
            
            other_df = ps.DataFrame(
                data=other_transposed,
                index=index_names,
            )
            
            result_df = self.data.dot(other_df)
            
            if other.shape[1] == self.shape[1]:
                new_cols = list(self.columns)
                if len(new_cols) == len(result_df.columns):
                    result_df.columns = new_cols

        elif isinstance(other, SparkDataset):
            common_cols = self.data.columns.intersection(other.data.columns)
            if len(common_cols) == 0:
                raise ValueError("Columns must match for SparkDataset.dot")
            
            product = self.data[common_cols] * other.data[common_cols]
            
            if self.shape[0] == other.shape[0] and len(common_cols) == 1:
                result_series = product.sum() 
                if isinstance(result_series, (int, float, np.number)):
                    result_df = ps.DataFrame([[result_series]], columns=["dot_product"])
                else:
                    result_df = result_series.to_frame().T
            else:
                result_df = product.sum()
                if isinstance(result_df, ps.Series):
                    result_df = result_df.to_frame().T

        else:
            raise TypeError(f"Unsupported type for dot: {type(other)}")

        if not isinstance(result_df, ps.DataFrame):
            result_df = ps.DataFrame(result_df)

        return self.__class__(data=result_df, session=self.session)


    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ) -> SparkDataset:
        result = self.data.dropna(how=how, subset=subset, axis=axis)
        return self.__class__(data=result, session=self.session)

    def transpose(self, names: Sequence[str] | None = None) -> SparkDataset:
        result = self.data.transpose()
        if names is not None:
            result.columns = names
        return self.__class__(data=result, session=self.session) if isinstance(result, ps.DataFrame) else self.__class__(data=ps.DataFrame(result), session=self.session)

    def sample(self,
               frac: float | None = None,
               n: int | None = None,
               random_state: int | None = None) -> SparkDataset:
        if n is not None:
            if frac is not None:
                raise ValueError("Cannot specify both 'n' and 'frac'")
            
            total_rows = len(self.data)
            if total_rows == 0:
                frac = 0.0
            else:
                frac = n / total_rows
        
        if frac is None and n is None:
            frac = 1.0
        
        result = self.data.sample(frac=frac, random_state=random_state)
        
        return self.__class__(data=result, session=self.session)

    def select_dtypes(
        self,
        include: str | None = None,
        exclude: str | None = None,
    ) -> SparkDataset:
        result = self.data.select_dtypes(include=include, exclude=exclude)
        return self.__class__(data=result, session=self.session)

    def isin(self, values: Iterable) -> SparkDataset:
        result = self.data.isin(values)
        return self.__class__(data=result, session=self.session)

    def merge(
        self,
        right: SparkDataset,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool | None = None,
        right_index: bool | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> SparkDataset:
        for on_ in [on, left_on, right_on]:
            if on_ and (
                on_ not in [*self.columns, *right.columns]
                if isinstance(on_, str)
                else any(c not in [*self.columns, *right.columns] for c in on_)
            ):
                raise MergeOnError(on_)
        if not all([on, left_on, right_on,]) and all([left_index is None, right_index is None]):
            left_index = True
            right_index = True
        result = self.data.merge(
            right=right.data,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            how=how,
        )
        return self.__class__(data=result, session=self.session)

    def drop(
        self,
        labels: str | None = None,
        axis: int | None = None,
        columns: str | Iterable[str] | None = None,
    ) -> SparkDataset:
        result = self.data.drop(labels=labels, axis=axis, columns=columns)
        return self.__class__(data=result, session=self.session)

    def filter(
        self,
        items: list | None = None,
        regex: str | None = None,
        axis: int = 0,
    ) -> SparkDataset:
        if items is not None:
            result = self.data[items]
        elif regex is not None:
            import re
            pattern = re.compile(regex)
            matched_cols = [col for col in self.data.columns if pattern.match(col)]
            result = self.data[matched_cols]
        else:
            if axis == 1:
                result = self.data
            else:
                result = self.data
        return self.__class__(data=result, session=self.session)

    def rename(self, columns: dict[str, str]) -> SparkDataset:
        result = self.data.rename(columns=columns)
        return self.__class__(data=result, session=self.session)

    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> SparkDataset:
        if isinstance(to_replace, ps.DataFrame) and len(to_replace.columns) == 1:
            to_replace = to_replace.iloc[:, 0]
        elif isinstance(to_replace, ps.Series):
            to_replace = to_replace.to_list()
        elif isinstance(to_replace, dict):
            result = self.data.replace(to_replace=to_replace, regex=regex)
        else:
            result = self.data.replace(to_replace=to_replace, value=value, regex=regex)
        return self.__class__(data=result, session=self.session)

    def reindex(self, labels: str = "", fill_value: str | None = None) -> SparkDataset:
        result = self.data.reindex(labels, fill_value=fill_value)
        return self.__class__(data=result, session=self.session)
    
    def fillna(self,
               values: ScalarType | dict[str, ScalarType] | None = None,
               method: Literal["bfill", "ffill"] | None = None,
               **kwargs) -> SparkDataset:
        if method is not None:
            if method == "bfill":
                result = self.data.bfill(**kwargs)
            elif method == "ffill":
                result = self.data.ffill(**kwargs)
            else:
                raise ValueError(f"Wrong fill method: {method}")
        else:
            result = self.data.fillna(value=values, **kwargs)
        return self.__class__(data=result, session=self.session)

    def list_to_columns(self, column: str) -> SparkDataset:
        data = self.data
        n_cols = len(data.loc[0, column]) 
        data_expanded = (
            ps.DataFrame(
                data[column].to_list(), columns=[f"{column}_{i}" for i in range(n_cols)]
            )
            if n_cols > 1
            else data
        )
        return self.__class__(data=data_expanded, session=self.session)