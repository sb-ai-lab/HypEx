from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, Sized, Self

import numpy as np
import pandas as pd

import pyspark.sql as spark
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDF
import pyspark.sql.functions as F
from pyspark.sql.types import StructType


import pyspark.pandas as ps
from pyspark.pandas.exceptions import PandasNotImplementedError

ps.set_option('compute.ops_on_diff_frames', True)
        
from ...utils import FromDictTypes, MergeOnError, ScalarType, SparkTypeMapper
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    PANDAS_CONVERSION_LIMIT: int = 100_000

    def checkpoint(self):
        pass
    
    def limit(self, num: int | None = None) -> Any:
        return self._wrap_result(self.data.iloc[:num])
    
    def _check_pandas_conversion(self, obj: ps.DataFrame | ps.Series, context: str = "") -> None:
        n: int = obj.__len__()
        if n > self.PANDAS_CONVERSION_LIMIT:
            raise ValueError(f"{context}: {n} rows exceed limit {self.PANDAS_CONVERSION_LIMIT}")
        
    def _wrap_result(self, 
                     result: ps.DataFrame | ps.Series | Any,
                     wrap_series: bool = False) -> Self | ps.Series | Any:
        if isinstance(result, ps.DataFrame):
            return self.__class__(data=result, session=self.session)
        
        if wrap_series and isinstance(result, ps.Series):
            return self.__class__(data=result.to_frame(), session=self.session)
        
        return result
    
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
        if isinstance(data, dict):
            if "index" in data:
                data = pd.DataFrame(data=data["data"], index=data["index"])
            else:
                data = pd.DataFrame(data=data["data"])

        if session is None:
            if isinstance(data, ps.DataFrame):
                session = data.to_spark().sparkSession
            elif isinstance(data, SparkDF):
                session = data.to_spark().sparkSession
            else:
                raise ValueError(
                    "Session must be provided explicitly or inferred from "
                    "ps.DataFrame/SparkDF data"
                )
        
        if not isinstance(session, SparkSession):
            raise TypeError("Session must be an instance of SparkSession")
        
        self.session = session

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

    def __getitem__(self, 
                    item: slice | int | str | list | ps.DataFrame | ps.Series) -> Self | ps.Series:        
        if isinstance(item, (slice, int)):
            return self._wrap_result(self.data.iloc[item])
        if isinstance(item, str):
            result = self.data[item]
            if isinstance(result, ps.Series):
                result = result.to_frame()
            return self._wrap_result(result)
        if isinstance(item, list):
            return self._wrap_result(self.data[item])
        if isinstance(item, ps.DataFrame):
            if len(item.columns) != 1:
                raise ValueError("Boolean DataFrame mask must have exactly one column")
            
            return self._wrap_result(self.data[item.iloc[:, 0]])
        if isinstance(item, ps.Series):
            return self._wrap_result(self.data[item])
        raise KeyError("No such column or row")

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def __magic_determine_other(other: Any) -> Any:
        if isinstance(other, SparkDataset):
            return other.data
        else:
            return other

    # comparison operators:
    def __eq__(self, other: Any) -> Self:
        return self._wrap_result(self.data == self.__magic_determine_other(other))

    def __ne__(self, other: Any) -> Self:
        return self._wrap_result(self.data != self.__magic_determine_other(other))

    def __le__(self, other: Any) -> Self:
        return self._wrap_result(self.data <= self.__magic_determine_other(other))

    def __lt__(self, other: Any) -> Self:
        return self._wrap_result(self.data < self.__magic_determine_other(other))

    def __ge__(self, other: Any) -> Self:
        return self._wrap_result(self.data >= self.__magic_determine_other(other))

    def __gt__(self, other: Any) -> Self:
        return self._wrap_result(self.data > self.__magic_determine_other(other))

    # unary operations:
    def __pos__(self) -> Self:
        return self._wrap_result(+self.data)

    def __neg__(self) -> Self:
        return self._wrap_result(-self.data)

    def __abs__(self) -> Self:
        return self._wrap_result(abs(self.data))

    def __invert__(self) -> Self:
        return self._wrap_result(~self.data)

    def __round__(self, ndigits: int = 0) -> Self:
        return self._wrap_result(self.data.round(ndigits))

    # Binary operations:
    def __add__(self, other: Any) -> Self:
        return self._wrap_result(self.data + self.__magic_determine_other(other))

    def __sub__(self, other: Any) -> Self:
        return self._wrap_result(self.data - self.__magic_determine_other(other))

    def __mul__(self, other: Any) -> Self:
        return self._wrap_result(self.data * self.__magic_determine_other(other))

    def __floordiv__(self, other: Any) -> Self:
        return self._wrap_result(self.data // self.__magic_determine_other(other))

    def __div__(self, other: Any) -> Self:
        return self._wrap_result(self.data / self.__magic_determine_other(other))

    def __truediv__(self, other: Any) -> Self:
        return self._wrap_result(self.data / self.__magic_determine_other(other))

    def __mod__(self, other: Any) -> Self:
        return self._wrap_result(self.data % self.__magic_determine_other(other))

    def __pow__(self, other: Any) -> Self:
        return self._wrap_result(self.data ** self.__magic_determine_other(other))

    def __and__(self, other: Any) -> Self:
        return self._wrap_result(self.data & self.__magic_determine_other(other))

    def __or__(self, other: Any) -> Self:
        return self._wrap_result(self.data | self.__magic_determine_other(other))

    # Right arithmetic operators:
    def __radd__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) + self.data)

    def __rsub__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) - self.data)

    def __rmul__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) * self.data)

    def __rfloordiv__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) // self.data)

    def __rdiv__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) / self.data)

    def __rtruediv__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) / self.data)

    def __rmod__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) % self.data)

    def __rpow__(self, other: Any) -> Self:
        return self._wrap_result(self.__magic_determine_other(other) ** self.data)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def _repr_html_(self) -> str:
        return self.data._repr_html_()
    
    def _display_head_tail(self, 
                           rows_display_limit: int,
                           cols_display_limit: int,
                           n_cols: int,
                           n_rows: int,  
                           tail: bool=False
    ) -> pd.DataFrame:
        """Returns n head rows or n tail rows
        Args:
            rows_display_limit: rows display limit.
            cols_display_limit: columns display limit.
            n_cols: number of columns in dataframe.
            n_rows: number of rows in dataframe.
            tail: flag of direction. If False, head rows returned.
        Return:
            pd.DataFrame: head or tail part of dataframe. 
        """
        if tail:
            head_tail = self.data.tail(rows_display_limit).to_pandas()
            head_tail.index = [(n_rows - rows_display_limit + i) for i in range(rows_display_limit)]
        else:
            head_tail = self.data.head(rows_display_limit).to_pandas()

        if n_cols > 2 * cols_display_limit:
            left_cols = self.columns[: cols_display_limit]
            right_cols = self.columns[-cols_display_limit:]
            tmp = pd.DataFrame(
                [["..."] for _ in range(len(head_tail))],
                index=head_tail.index,
                columns=["..."]
            )

            return pd.concat([head_tail.loc[:, left_cols],
                              tmp,
                              head_tail.loc[:, right_cols]],
                             axis=1)
        else:
            return head_tail

    def get_values(self, 
                   row: str | None = None, 
                   column: str | None = None) -> ScalarType | Sequence[ScalarType] | Self | ps.Series:
        if (column is not None) and (row is not None):
            return self._wrap_result(self.data.loc[row, column])
        elif column is not None:
            result = self.data.loc[:, column]
        elif row is not None:
            result = self.data.loc[row, :]
        else:
            result = self.data
        
        if isinstance(result, (ps.DataFrame, ps.Series)):
            self._check_pandas_conversion(obj=result, context="get_values")
            return self._wrap_result(result.to_pandas().values.tolist())
        return self._wrap_result(result)

    def iget_values(self, 
                    row: int | None = None, 
                    column: int | None = None) -> ScalarType | Sequence[ScalarType] | Self | ps.Series:
        if (column is not None) and (row is not None):
            return self._wrap_result(self.data.iloc[row, column])
        elif column is not None:
            result = self.data.iloc[:, column]
        elif row is not None:
            result = self.data.iloc[row, :]
        else:
            result = self.data
            
        if isinstance(result, (ps.DataFrame, ps.Series)):
            self._check_pandas_conversion(obj=result, context="iget_values")
            return self._wrap_result(result.to_pandas().values.tolist())
        return self._wrap_result(result)

    def create_empty(self, 
                     index: Iterable[Any] | None = None, 
                     columns: Iterable[str] | None = None) -> "SparkNavigation":
        return self._wrap_result(ps.DataFrame(index=index, columns=columns))

    @property
    def index(self) -> ps.Index:
        return self.data.index
    
    def reset_index(self, 
                    drop: bool = False,
                    inplace: bool = False,
                    **kwargs) -> Self | None:
        kwargs['inplace'] = False
        
        result = self.data.reset_index(drop=drop, **kwargs)
        return self._wrap_result(result)

    @property
    def columns(self) -> list[str]:
        return self.data.columns.tolist()

    # @property
    # def session(self):
    #     return self.session

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape
    
    @property
    def labels_dict(self):
        return {}

    def _get_column_index(self, column_name: Sequence[str] | str) -> int | list[int]:
        if isinstance(column_name, str):
            return self.data.columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return self.data.columns.get_indexer(column_name)
        else:
            raise ValueError("Wrong column_name type.")

    def get_column_type(self, column_name: str | Iterable[str] | None = None) -> dict[str, type] | type | None:
        spark_schema = self.data.to_spark().schema
        
        if isinstance(column_name, str):
            field = next((f for f in spark_schema.fields if f.name == column_name), None)
            return SparkTypeMapper.to_python(field.dataType) if field else None
        
        result = {}
        target_cols = column_name if column_name is not None else self.data.columns
        for col in target_cols:
            field = next((f for f in spark_schema.fields if f.name == col), None)
            result[col] = SparkTypeMapper.to_python(field.dataType) if field else object
        
        return result           

    def astype(self, 
               dtype: dict[str, type], 
               errors: Literal["raise", "ignore"] = "raise") -> Self:
        return self._wrap_result(self.data.astype(dtype=dtype))

    def update_column_type(self,
                           dtype: dict[str, type],
                           errors: Literal["raise", "ignore"] = "raise") -> SparkNavigation:
        for column_name, target_type in dtype.items():
            if column_name not in self.data.columns:
                if errors == "raise":
                    raise KeyError(f"Column '{column_name}' not found")
                continue
            
            if self.data[column_name].isna().all():
                if errors == "raise":
                    raise ValueError(
                        f"Cannot infer type for column '{column_name}': all values are null"
                    )
                continue
            
            try:
                self.data = self.data.astype({column_name: target_type})
            except (ValueError, TypeError) as e:
                if errors == "raise":
                    raise type(e)(
                        f"Failed to convert column '{column_name}' to {target_type}: {e}"
                    )        
        return self._wrap_result(self.data)

    def add_column(self, 
                   data: Sequence[Any], 
                   name: str | list[str], 
                   index: Sequence[Any] | None = None) -> None:
        if isinstance(name, list) and len(name) == 1:
            name = name[0]
        if isinstance(data, (ps.DataFrame, ps.Series)):
            if isinstance(data, ps.DataFrame) and data.shape[1] == 1:
                data_col = data.columns[0]
                data_tmp = data
                if name != data_col:
                    data_tmp[name] = data_tmp[data_col]
                    data_tmp = data_tmp.drop(columns=[data_col])   
                self.data = self.data.join(data_tmp)
                return
            self.data = self.data.join(data)
            return
        
        self.data[name] = data

    def append(self, 
               other: Sequence[SparkNavigation], 
               reset_index: bool = False, 
               axis: int = 0) -> Self:
        new_data = ps.concat([self.data] + [d.data for d in other], axis=axis)
        if reset_index:
            new_data = new_data.reset_index(drop=True)
        return self._wrap_result(new_data)

    def from_dict(self, 
                  data: FromDictTypes, 
                  index: Iterable[Any] | Sized | None = None):
        if isinstance(data, dict):
            self.data = ps.DataFrame().from_records(data, columns=list(data.keys()))
        else:
            self.data = ps.DataFrame().from_records(data)
        if index is not None:
            self.data.index = index
        return self

    def to_dict(self) -> dict[str, list[Any]]:
        self._check_pandas_conversion(obj=self.data, context="to_dict")
        pdf = self.data.to_pandas()
        return {
            "data": {
                column: pdf[column].to_list() for column in pdf.columns
            },
            "index": list(pdf.index),
        }

    def to_records(self) -> list[dict[str, Any]]:
        self._check_pandas_conversion(obj=self.data, context="to_records")
        return self.data.to_pandas().to_dict(orient="records")

    def loc(self, items: Iterable[Any]) -> Self:
        data = self.data.loc[items]
        return data

    def iloc(self, items: Iterable[Any]) -> Self:
        if isinstance(items, int): 
            data = self.data.iloc[[items]]
        else:
            data = self.data.iloc[items]
        if not isinstance(data, ps.DataFrame):
            data = ps.DataFrame(data)
        return data


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    @staticmethod
    def _convert_agg_result(result: ps.Series | ps.DataFrame) -> Self | float:
        if isinstance(result, ps.Series):
            result = result.to_frame()
        if result.shape == (1, 1):
            return float(result.to_spark().collect()[0][0])
        return result if isinstance(result, ps.DataFrame) else ps.DataFrame(result)

    def __init__(self, 
                 data: ps.DataFrame | dict | str | ps.Series | None = None,
                 session: SparkSession | None = None):
        super().__init__(data=data, session=session)

    def get(self, key: str, default: Any=None) -> Any:
        return self.data.get(key, default)

    def take(self, 
             indices: int | Sequence[int], 
             axis: Literal["index", "columns", "rows"] | int = 0) -> Self | ps.Series:
        if isinstance(indices, slice) and (axis == 1):
            self._wrap_result(self.data.iloc[indices])
        return self._wrap_result(self.data.take(indices=indices, axis=axis))

    def apply(self, func: Callable[..., Any], **kwargs) -> SparkDataset:
        single_column_name = kwargs.pop("column_name", None)
        result = self.data.apply(func, **kwargs)
        if not isinstance(result, ps.DataFrame):
            result = result.to_frame(name=single_column_name)
        return self._wrap_result(result)

    def map(self, func: Callable[..., Any], na_action: Any = None, **kwargs) -> SparkDataset:
        return self._wrap_result(self.data.map(func, **kwargs))

    def is_empty(self) -> bool:
        return self.data.empty

    def unique(self) -> dict[str, list[Any]]:
        return {column: self.data[column].unique() for column in self.data.columns}

    def nunique(self, dropna: bool = True)-> dict[str, int]:
        return {column: self.data[column].nunique() for column in self.data.columns}
    
    def groupby(self, by: str | Iterable[str], **kwargs) -> ps.groupby.GroupBy:
        return self.data.groupby(by=by, **kwargs)

    def agg(self, func: str | list, **kwargs) -> SparkDataset | float:
        subset = kwargs.pop('subset', None)
        func = func if isinstance(func, (list, dict)) else [func]
        
        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            data_to_agg = self.data[subset]
        else:
            types = self.get_column_type()
            numeric_cols = [
                col for col, dtype in types.items() 
                if dtype in [int, float, np.int64, np.float64, np.int32, np.float32]
            ]
            
            if len(numeric_cols) == 0:
                return None
            
            data_to_agg = self.data[numeric_cols]
        
        if data_to_agg is None or len(data_to_agg.columns) == 0:
            return None
        
        if isinstance(func, list) and len(func) == 1:
            agg_dict = {col: func[0] for col in data_to_agg.columns}
        else:
            agg_dict = {col: func for col in data_to_agg.columns}
        
        result = data_to_agg.agg(agg_dict, **kwargs)
        converted = self._convert_agg_result(result)
        
        if isinstance(converted, ps.DataFrame):
            return self._wrap_result(converted)
        return self._wrap_result(converted)

    def max(self) -> SparkDataset | float:
        return self._wrap_result(self.agg(["max"]))

    def idxmax(self) -> SparkDataset | float:
        return self._wrap_result(self.agg(["idxmax"]))

    def min(self) -> SparkDataset | float:
        return self._wrap_result(self.agg(["min"]))

    def count(self) -> SparkDataset | float:
        return self._wrap_result(self.agg(["count"]))

    def sum(self) -> SparkDataset | float:
        return self._wrap_result(self.agg(["sum"]))

    def mean(self) -> SparkDataset | float:
        return self._wrap_result(self.agg("mean"))

    def mode(self, numeric_only: bool = False, dropna: bool = True) -> SparkDataset:
        return self._wrap_result(self.data.mode(numeric_only=numeric_only, dropna=dropna))

    def std(self, skipna: bool = True, ddof: int = 1) -> Self | float:
        result = self.data.std()
        converted = self._convert_agg_result(result.to_frame() if isinstance(result, ps.Series) else result)
        return self._wrap_result(converted)

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> SparkDataset | float:
        result = self.data.var()
        converted = self._convert_agg_result(result.to_frame() if isinstance(result, ps.Series) else result)
        return self._wrap_result(converted)

    def log(self) -> SparkDataset:
        np_data = np.log(self.data.to_numpy())
        return self._wrap_result(ps.DataFrame(np_data, columns=self.data.columns))

    def cov(self) -> SparkDataset:
        numeric_cols = self.get_numeric_columns()        
        if len(numeric_cols) == 0:
            return None
        
        result = self.data[numeric_cols].cov()
        return self._wrap_result(result)

    def quantile(self, q: float = 0.5) -> Self | float:
        if isinstance(q, list) and len(q) > 1:
            return self.data.quantile(q=q)
        else:
            result = self.data.quantile(q=q)
            converted = self._convert_agg_result(result.to_frame() if isinstance(result, ps.Series) else result)
            if isinstance(converted, ps.DataFrame):
                return converted
            return self._wrap_result(converted)

    def coefficient_of_variation(self) -> Self | float:
        numeric_cols = self.get_numeric_columns()
        if len(numeric_cols) == 0:
            return None
        
        data_to_calc = self.data[numeric_cols]
        
        std_series = data_to_calc.std()
        mean_series = data_to_calc.mean()
        
        cv_series = std_series / mean_series.replace(0, np.nan)
        cv_df = cv_series.to_frame().T
        
        if cv_df.shape[0] == 1 and cv_df.shape[1] == 1:
            return float(cv_df.to_spark().collect()[0][0])
        
        try:
            old_index_name = cv_df.index.tolist()[0]
            cv_df = cv_df.rename(index={old_index_name: "cv"})
        except (PandasNotImplementedError, AttributeError):
            cv_df = cv_df.rename(index={0: "cv"})
        
        return self._wrap_result(cv_df)

    def sort_index(self, ascending: bool = True, **kwargs) -> Self:
        return self._wrap_result(self.data.sort_index(ascending=ascending, **kwargs))

    def get_numeric_columns(self) -> list[str]:
        types = self.get_column_type()
        return [col for col, dtype in types.items() if dtype in [int, float, np.int64, np.float64, np.int32, np.float32]]

    def corr(self, numeric_only: bool = False) -> Self | float:
        numeric_cols = self.get_numeric_columns()
        
        if len(numeric_cols) == 0:
            return None
        
        result = self.data[numeric_cols].corr(method='pearson')
        
        if isinstance(result, ps.DataFrame):
            return self._wrap_result(result)
        return result

    def isna(self) -> Self:
        return self._wrap_result(self.data.isna())

    def sort_values(self, by: str | list[str], ascending: bool = True, **kwargs) -> Self:
        return self._wrap_result(self.data.sort_values(by=by, ascending=ascending, **kwargs))

    def value_counts(self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True) -> Self:
        
        col = list(self.data.columns)[0]
        series = self.data[col]
        
        result = series.value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna
        )
        
        result_df = result.to_frame(name="count").reset_index()
        
        result_df = result_df.rename(columns={"index": col})
        
        return self._wrap_result(result_df)

    def na_counts(self) -> Self | int:
        data = self.data.isna().sum().to_frame().T
        
        if data.shape[0] == 1 and data.shape[1] == 1:
            return int(data.to_spark().collect()[0][0])
        
        old_index_name = data.index.tolist()[0]
                
        return self._wrap_result(data.rename(index={old_index_name: "na_counts"}))

    def dot(self, other: 'SparkDataset' | np.ndarray | pd.DataFrame) -> Self | float:
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) != len(self.data.columns):
                    raise ValueError(
                        f"Vector length ({len(other)}) must match number of columns ({len(self.data.columns)})"
                    )
                other_series = ps.Series(other, index=self.data.columns)
                result = self.data.dot(other_series)
            else:
                other_df = ps.DataFrame(other)
                if other_df.shape[0] != len(self.data.columns):
                    raise ValueError(
                        f"Matrix dimensions not aligned: {self.data.shape} dot {other_df.shape}"
                    )
                other_df.index = self.data.columns
                result = self.data.dot(other_df)
            return self._wrap_result(result if isinstance(result, ps.DataFrame) else result.to_frame())
        
        elif isinstance(other, pd.DataFrame):
            other_ps = ps.DataFrame(other)
            if other_ps.shape[0] != len(self.data.columns):
                raise ValueError(
                    f"Matrix dimensions not aligned: {self.data.shape} dot {other_ps.shape}"
                )
            other_ps.index = self.data.columns
            result = self.data.dot(other_ps)
            return self._wrap_result(result if isinstance(result, ps.DataFrame) else result.to_frame())
        
        elif isinstance(other, SparkDataset):
            common_cols = self.data.columns.intersection(other.data.columns)
            
            if len(common_cols) == 0:
                raise ValueError(
                    f"No common columns for dot product. "
                    f"Self columns: {self.columns}, Other columns: {other.columns}"
                )
            
            other_subset = other.data[common_cols]
            self_subset = self.data[common_cols]
            
            if len(common_cols) == 1:
                result = self_subset.iloc[:, 0].dot(other_subset.iloc[:, 0])
                return self._wrap_result(float(result) if isinstance(result, (int, float, np.number)) else result)
            else:
                result = (self_subset * other_subset).sum()
                return self._wrap_result(result if isinstance(result, ps.DataFrame) else result.to_frame())
        
        else:
            raise TypeError(
                f"Unsupported type for dot: {type(other)}. "
                f"Expected SparkDataset, np.ndarray, or pd.DataFrame"
            )


    def dropna(self,
               how: Literal["any", "all"] = "any",
               subset: str | Iterable[str] | None = None,
               axis: Literal["index", "rows", "columns"] | int = 0) -> SparkDataset:
        return self._wrap_result(self.data.dropna(how=how, subset=subset, axis=axis))

    def transpose(self, names: Sequence[str] | None = None) -> Self:
        result = self.data.transpose()
        if names is not None:
            result.columns = names
        return self._wrap_result(result if isinstance(result, ps.DataFrame) else ps.DataFrame(result))
    
    @staticmethod
    def _reproducible_sample(df: ps.DataFrame,
                             n: int = None,
                             frac: float = None,
                             replace: bool = False,
                             seed: int = 42) -> ps.DataFrame:
        
        df_with_index = df.reset_index()
        index_cols = df_with_index.columns[:df.index.nlevels]
        
        df_with_index['_shuffle_key'] = F.abs(
            F.hash(*[F.col(col) for col in index_cols], F.lit(seed))
        )
        
        shuffled = df_with_index.sort_values('_shuffle_key')
        
        total_rows = len(df)
        if n is not None:
            sample_size = n
        elif frac is not None:
            sample_size = int(total_rows * frac)
        else:
            sample_size = total_rows
        
        if replace:
            indices = [i % total_rows for i in range(sample_size)]
            sampled = shuffled.iloc[indices]
        else:
            sampled = shuffled.iloc[:sample_size]
        
        sampled = sampled.drop(columns=['_shuffle_key'])
        if df.index.nlevels == 1:
            sampled = sampled.set_index(index_cols[0])
        else:
            sampled = sampled.set_index(index_cols)
        
        return sampled

    def sample(self,
               frac: float | None = None,
               n: int | None = None,
               random_state: int | None = None,
               method: Literal["approx", "exact"] = "exact") -> Self:
        
        # if n is not None and frac is not None:
        #     raise ValueError("Cannot specify both 'n' and 'frac'")
        
        # spark_df = self.data.to_spark()
        
        # if n is not None:
        #     total = spark_df.count()
        #     if n >= total:
        #         return self._wrap_result(self.data)
            
        #     if method == "exact":
        #         sampled = spark_df.orderBy(F.rand(seed=random_state)).limit(n)
        #     else:
        #         frac_calc = min(1.0, n / total * 1.3)
        #         sampled = spark_df.sample(
        #             withReplacement=False, 
        #             fraction=frac_calc, 
        #             seed=random_state
        #         ).limit(n)
            
        #     return self._wrap_result(ps.DataFrame(sampled))
        
        # return self._wrap_result(self.data.sample(frac=frac or 1.0, random_state=random_state))
        return self._wrap_result(self._reproducible_sample(df=self.data, n=n, frac=frac or 1.0, seed=random_state))


    def select_dtypes(self,
                      include: str | None = None,
                      exclude: str | None = None) -> Self:
        return self._wrap_result(self.data.select_dtypes(include=include, exclude=exclude))

    def isin(self, values: Iterable) -> SparkDataset:
        return self._wrap_result(self.data.apply(lambda col: col.isin(values)))

    def merge(self,
              right: SparkDataset,
              on: str | None = None,
              left_on: str | None = None,
              right_on: str | None = None,
              left_index: bool | None = None,
              right_index: bool | None = None,
              suffixes: tuple[str, str] = ("_x", "_y"),
              how: Literal["left", "right", "inner", "outer", "cross"] = "inner") -> SparkDataset:
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
        return self._wrap_result(result)

    def drop(self,
             labels: str | None = None,
             axis: int | None = None,
             columns: str | Iterable[str] | None = None) -> Self:
        return self._wrap_result(self.data.drop(labels=labels, axis=axis, columns=columns))

    def filter(self,
               items: list | None = None,
               regex: str | None = None,
               axis: int | str = 0) -> Self:
        return self._wrap_result(self.data.filter(items=items, regex=regex, axis=axis))

    def rename(self, columns: dict[str, str]) -> Self:
        return self._wrap_result(self.data.rename(columns=columns))

    def replace(self, to_replace: Any = None, value: Any = None, regex: bool = False) -> Self:
        if isinstance(to_replace, ps.DataFrame) and len(to_replace.columns) == 1:
            to_replace = to_replace.iloc[:, 0]
        elif isinstance(to_replace, ps.Series):
            to_replace = to_replace.to_list()
        elif isinstance(to_replace, dict):
            result = self.data.replace(to_replace=to_replace, regex=regex)
        else:
            result = self.data.replace(to_replace=to_replace, value=value, regex=regex)
        return self._wrap_result(result)
    

    def reindex(self, labels: str = "", fill_value: str | None = None) -> SparkDataset:
        return self._wrap_result(self.data.reindex(labels, fill_value=fill_value))
    
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
        return self._wrap_result(result)

    def list_to_columns(self, column: str) -> Self:
        data = self.data
        n_cols = len(data.loc[0, column]) 
        data_expanded = (
            ps.DataFrame(
                data[column].to_list(), columns=[f"{column}_{i}" for i in range(n_cols)]
            )
            if n_cols > 1
            else data
        )
        return data_expanded