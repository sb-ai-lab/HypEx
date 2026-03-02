from __future__ import annotations

import os
import sys
import re
from itertools import chain
from functools import reduce
from contextlib import contextmanager

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
)

import numpy as np
import pandas as pd
import pyspark.sql as spark
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import Window, SparkSession, Row, Column
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame as SparkDF

from pyspark.sql.types import (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
    NumericType,
    
    StringType,
    
    BooleanType,
    
    DateType,
    TimestampType,
    
    ArrayType,
    StructType,
    StructField
)

from ...utils import FromDictTypes, MergeOnError, ScalarType, Adapter
from ...utils.typings import SparkTypeMapper as stm
from ...utils.constants import UTILITY_INDEX_COL_NAME, UTILITY_PHYSICAL_INDEX_COL_NAME
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(filename: str | Path, session: SparkSession) -> SparkDF:
        file_path = Path(filename).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: '{file_path}'")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: '{file_path}'")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied: '{file_path}'")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".csv":
            return (session
                .read
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("encoding", "UTF-8")
                .load(str(file_path))
            )
        elif suffix == ".parquet":
            return session.read.parquet(str(file_path))
        elif suffix == ".json":
            return session.read.json(str(file_path))
        elif suffix == ".orc":
            return session.read.orc(str(file_path))
        else:
            raise ValueError(f"Unsupported file extension: '{suffix}'. "
                             f"Supported: .csv, .parquet, .json, .orc")            


    def __init__(self,
                 data: SparkDF | pd.DataFrame | dict[str, Any] | str | None = None,
                 session: SparkSession | None = None,
                 physical_index_actual_flag: bool = True):
        if session is None:
            raise ValueError("Session not set")
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

        if isinstance(data, SparkDF):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.data = self.session.createDataFrame(data)
        elif isinstance(data, str):
            self.data = self._read_file(data, self.session)
        else:
            self.data = self.session.createDataFrame([], schema=StructType([]))
            
        if UTILITY_INDEX_COL_NAME not in self.data.columns:
            self.data = self.__add_row_index(df=self.data, index_column_name=UTILITY_INDEX_COL_NAME)
            self.data = self.data.withColumn(UTILITY_PHYSICAL_INDEX_COL_NAME, F.col(UTILITY_INDEX_COL_NAME))
            
        self._physical_index_actual_flag: bool = physical_index_actual_flag
        self._count_data: int | None = None
        
    @property
    def _public_columns(self) -> list[str]:
        return [c for c in self.data.columns 
                if c not in (UTILITY_INDEX_COL_NAME, UTILITY_PHYSICAL_INDEX_COL_NAME)]
            
    def __add_row_index(self, df: SparkDF, index_column_name: str) -> SparkDF:
        rdd = df.rdd.zipWithIndex()
        new_rdd = rdd.map(lambda x: tuple(x[0]) + (x[1],))
        new_schema = T.StructType(df.schema.fields + [T.StructField(index_column_name, T.LongType(), False)])
        return self.session.createDataFrame(new_rdd, schema=new_schema)
   
    def __reindex_physical_index(self) -> None:
        self.data = self.__add_row_index(
            df=self.data.drop(F.col(UTILITY_PHYSICAL_INDEX_COL_NAME)).orderBy(F.col(UTILITY_INDEX_COL_NAME)), 
            index_column_name=UTILITY_PHYSICAL_INDEX_COL_NAME
        ) 

    def __len__(self) -> int:
        if self.count_data is None:
            self._count_data = 0 if self.data is None else self.data.count()
        return self._count_data
    
    @staticmethod
    def __magic_determine_other(
        other: 'SparkDataset' | 'SparkNavigation' | SparkDF | 
               Column | list[int | float | str | bool] | tuple | 
               int | float | str | 
               bool | np.generic | np.ndarray) -> list[Column] | Column:
        if isinstance(other, (SparkDataset, SparkNavigation)):
            return [F.col(c) for c in other.data.columns 
                    if c not in (UTILITY_INDEX_COL_NAME, UTILITY_PHYSICAL_INDEX_COL_NAME)]
        elif isinstance(other, SparkDF):
            raise TypeError(
                "SparkDF cannot be used directly in operations. "
                "Wrap it in SparkDataset or extract columns explicitly."
            )
        elif isinstance(other, Column):
            return other
        elif isinstance(other, (list, tuple)):
            return [F.lit(v) for v in other]
        elif isinstance(other, (int, float, str, bool)):
            return F.lit(other)
        elif isinstance(other, np.generic):
            if isinstance(other, np.bool_):
                return F.lit(bool(other))
            return F.lit(other.item())
        elif isinstance(other, np.ndarray):
            if other.ndim != 1:
                raise ValueError(f"Only 1D numpy arrays supported, got {other.ndim}D")
            return [F.lit(v) for v in other.tolist()]
        else:
            raise TypeError(
                f"Unsupported operand type: '{type(other).__name__}'. "
                f"Expected: Dataset, Column, scalar, or list"
            )
        
    def add_column(self, data: SparkDF | list, name: str | None = None, index = None) -> 'SparkNavigation' | 'SparkDataset':
        raise NotImplementedError("For add column need implement uniq_column")

    # comparison operators:
    def __eq__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) == other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __ne__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) != other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __lt__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) < other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __le__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) <= other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __ge__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) >= other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __gt__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) > other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    # unary operators:
    def __pos__(self) -> 'SparkNavigation' | 'SparkDataset':
        result_df = self.data.select([F.col(c).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __neg__(self) -> 'SparkNavigation' | 'SparkDataset':
        result_df = self.data.select([(-F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __abs__(self) -> 'SparkNavigation' | 'SparkDataset':
        result_df = self.data.select([F.abs(F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __invert__(self) -> 'SparkNavigation' | 'SparkDataset':
        result_df = self.data.select([(~F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __round__(self, n: int = 0) -> 'SparkNavigation' | 'SparkDataset':
        result_df = self.data.select([F.round(F.col(c), n).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    # binary arithmetic operations:
    def __add__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) + other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __sub__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) - other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __mul__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) * other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __floordiv__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.floor(F.col(c) / other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __truediv__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) / other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)
    
    def __div__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        return self.__truediv__(other)

    def __mod__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) % other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __pow__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.pow(F.col(c), other_val).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    # binary logical operators:
    def __and__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        if isinstance(other_val, list):
            if len(other_val) != len(self._public_columns):
                raise ValueError(
                    f"Datasets must have the same number of columns for logical operations. "
                    f"Self: {len(self._public_columns)}, Other: {len(other_val)}"
                )
            result_df = self.data.select([
                (F.col(self_col) & other_val[i]).alias(self_col)
                for i, self_col in enumerate(self._public_columns)
            ])
        else:
            result_df = self.data.select([
                (F.col(c) & other_val).alias(c) for c in self._public_columns
            ])
        return self.__class__(data=result_df, session=self.session)

    def __or__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        if isinstance(other_val, list):
            if len(other_val) != len(self._public_columns):
                raise ValueError(
                    f"Datasets must have the same number of columns for logical operations. "
                    f"Self: {len(self._public_columns)}, Other: {len(other_val)}"
                )
            result_df = self.data.select([
                (F.col(self_col) | other_val[i]).alias(self_col)
                for i, self_col in enumerate(self._public_columns)
            ])
        else:
            result_df = self.data.select([
                (F.col(c) | other_val).alias(c) for c in self._public_columns
            ])
        return self.__class__(data=result_df, session=self.session)

    def __xor__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        if isinstance(other_val, list):
            if len(other_val) != len(self._public_columns):
                raise ValueError(
                    f"Datasets must have the same number of columns for logical operations. "
                    f"Self: {len(self._public_columns)}, Other: {len(other_val)}"
                )
            result_df = self.data.select([
                ((F.col(self_col) & ~other_val[i]) | (~F.col(self_col) & other_val[i])).alias(self_col)
                for i, self_col in enumerate(self._public_columns)
            ])
        else:
            try:
                result_df = self.data.select([F.xor(F.col(c), other_val).alias(c) for c in self._public_columns])
            except (AttributeError, TypeError):
                result_df = self.data.select([
                    ((F.col(c) & ~other_val) | (~F.col(c) & other_val)).alias(c)
                    for c in self._public_columns
                ])
        return self.__class__(data=result_df, session=self.session)

    # reverse arithmetic operators:
    def __radd__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        return self.__add__(other)

    def __rsub__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(other_val - F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __rmul__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        return self.__mul__(other)

    def __rfloordiv__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.floor(other_val / F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __rtruediv__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(other_val / F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __rdiv__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        return self.__rtruediv__(other)

    def __rmod__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(other_val % F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    def __rpow__(self, other) -> 'SparkNavigation' | 'SparkDataset':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.pow(other_val, F.col(c)).alias(c) for c in self._public_columns])
        return self.__class__(data=result_df, session=self.session)

    # representation methods:
    def __repr__(self) -> str:
        if self.data is None:
            return "SparkNavigation(data=None)"
        
        rows = self.data.count()
        columns = len(self._public_columns)
        schema_summary = ", ".join(f"{field.name}:{field.dataType.simpleString()}" 
                                   for field in self.data.schema.fields[:5])
        if len(self.data.schema.fields) > 5:
            schema_summary += f", ... ({len(self.data.schema.fields)} total)"
        
        return f"SparkNavigation(rows={rows}, columns={columns}, schema=[{schema_summary}])"

    def _repr_html_(self) -> str:
        if self.data is None:
            return "<p>SparkNavigation(data=None)</p>"
        try:
            pdf = self.data.limit(10).toPandas()
            html_table = pdf.to_html(index=False, classes="dataframe", border=0)
            
            rows = self.data.count()
            columns = len(self._public_columns)
            
            html_info = f"""
            <div style="margin: 10px 0;">
                <p style="font-family: monospace; font-size: 12px; color: #666;">
                    SparkNavigation: {rows} rows × {columns} columns
                </p>
            </div>
            {html_table}
            <div style="margin: 10px 0;">
                <p style="font-family: monospace; font-size: 12px; color: #666;">
                    [Showing first 10 rows of {rows}]
                </p>
            </div>
            """
            return html_info
        except Exception as e:
            return f"<p>Error rendering DataFrame: {str(e)}</p>"
    

    def get(self, key: str | int | list[Any] | tuple[Any, ...], default: Any = None) -> 'SparkNavigation | Any':
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_name = key
            if isinstance(row_idx, int):
                raise NotImplementedError("Access by row index is not supported")
        
        if not isinstance(key, (list, tuple)):
            key_list = [key]
        else:
            key_list = list(key)
            
        if not key_list:
            return default
            
        if isinstance(key_list[0], str):
            existing_cols = [k for k in key_list if k in self._public_columns]
            if existing_cols:
                return self.__class__(data=self.data.select(*existing_cols), session=self.session)
            else:
                return default
        elif isinstance(key_list[0], int):
            raise NotImplementedError("Access by row index list is not supported")
        else:
            return default

    def take(self, indices: int | list[int], axis: Literal["index", "columns", "rows"] | int = 0) -> 'SparkNavigation' | 'SparkDataset':
        if not isinstance(indices, list):
            indices = [indices]

        if axis in (1, "columns"):
            col_names = []
            for idx in indices:
                if 0 <= idx < len(self._public_columns):
                    col_names.append(self._public_columns[idx])
                else:
                    raise IndexError(f"Column index {idx} out of range")
            
            return self.__class__(data=self.data.select(*col_names), session=self.session)

        elif axis in (0, "index", "rows"):
            raise NotImplementedError("Taking rows by index is not supported in Spark")
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, 'index', 'columns', or 'rows'")

    def create_empty(self,
                     index: Iterable | None = None,
                     columns: Iterable[str] | None = None) -> "SparkNavigation":
        
        columns = list(columns) if columns is not None else []
        index_values = list(index) if index is not None else None
        
        if index_values is not None and len(index_values) > 0:
            rows = [
                (idx_val,) + tuple([None] * len(columns))
                for idx_val in index_values
            ]
            
            if index_values[0] is not None:
                first = index_values[0]
                if isinstance(first, bool):
                    index_type = T.BooleanType()
                elif isinstance(first, int):
                    index_type = T.LongType()
                elif isinstance(first, float):
                    index_type = T.DoubleType()
                elif isinstance(first, str):
                    index_type = T.StringType()
                else:
                    index_type = T.StringType()
            else:
                index_type = T.StringType()
            
            fields = [T.StructField("index", index_type, True)]
            fields += [T.StructField(col, T.StringType(), True) for col in columns]
            schema = T.StructType(fields)
            
            new_df = self.session.createDataFrame(rows, schema)
        
        elif columns:
            schema = T.StructType(
                [T.StructField(col, T.StringType(), True) for col in columns]
            )
            new_df = self.session.createDataFrame(
                self.session.sparkContext.emptyRDD(),
                schema
            )
        else:
            new_df = self.session.createDataFrame([], schema=T.StructType([]))
        
        return self.__class__(data=new_df, session=self.session)

    @property
    def columns(self) -> list[str]:
        return self._public_columns if self.data else []

    @property
    def shape(self) -> tuple[int, int]:
        if self.data:
            count = self.data.count()
            cols = len(self._public_columns)
            return (count, cols)
        return (0, 0)

    def _get_column_index(self, column_name: Sequence[str] | str) -> int | Sequence[int]:
        pd_index_columns = pd.Index(self._public_columns)
        if isinstance(column_name, str):
            return pd_index_columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return pd_index_columns.get_indexer(column_name)
        else:
            raise TypeError("Wrong column_name type.")

    def get_column_type(self, column_name: list[str] | str = None) -> dict[str, type] | type | None:
        if column_name is None:
            column_name = self._public_columns
        elif isinstance(column_name, str):
            column_name = [column_name]
        
        dtypes = {}
        for field in self.data.schema.fields:
            if field.name in column_name:
                dtypes[field.name] = stm.to_python(field.dataType)
        
        if isinstance(column_name, list):
            if len(column_name) == 1:
                return dtypes.get(column_name[0])
            return dtypes
        else:
            return dtypes.get(column_name[0]) if column_name else None

    def astype(self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise") -> 'SparkNavigation' | 'SparkDataset':
        for col, new_type in dtype.items():
            if col not in self._public_columns:
                if errors == "raise":
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                elif errors == "ignore":
                    continue
                else:
                    raise ValueError(f"Invalid errors parameter: {errors}")
            
            try:
                spark_type = stm.to_spark(new_type)
                new_df = self.data.withColumn(col, F.col(col).cast(spark_type))
            except Exception as e:
                if errors == "raise":
                    raise TypeError(f"Cannot cast column '{col}' to {new_type}: {e}")
                elif errors == "ignore":
                    continue
        
        return self.__class__(data=new_df, session=self.session)

    def update_column_type(self, dtype: dict[str, type]):
        if len(dtype) > 0:
            self.data = self.astype(dtype)
        return self

    def append(self, 
               other: 'SparkNavigation | list[SparkNavigation]',
               axis: int = 0) -> 'SparkNavigation' | 'SparkDataset':
        others = other if isinstance(other, list) else [other]
        
        datasets = [self.data] + [
            d.data if isinstance(d, SparkNavigation) else d 
            for d in others
        ]
        
        if axis == 0:
            new_data = reduce(
                lambda x, y: x.unionByName(y, allowMissingColumns=True), 
                datasets
            )
            return self.__class__(data=new_data, session=self.session, physical_index_actual_flag=False)
        
        if axis == 1:
            raise NotImplementedError("append on axis == 1 is not supported in Spark backend")
        
        raise ValueError("axis should be 0 or 1")
    
    @property
    def index(self):
        raise AttributeError("Spark-based Dataset has no index")
    
    def __getitem__(self, item: Any):
        raise NotImplementedError("Spark-base Dataset does not support indexing")

    def from_dict(self, data: FromDictTypes, index: Iterable | Sized | None = None):
        raise NotImplementedError("Need implement")

    def get_values(self, row: str | int | None = None, column: str | None = None) -> Any:
        raise NotImplementedError("Row-based value access is not supported")

    def iget_values(self, row: int | None = None, column: int | None = None) -> Any:
        raise NotImplementedError("Row-based value access is not supported")

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("to_dict is not supported in Spark backend")

    def to_records(self) -> list[dict]:
        raise NotImplementedError("to_records is not supported in Spark backend")

    def loc(self, items: Iterable) -> 'SparkNavigation' | 'SparkDataset':
        items_list = list(items) 
        
        if not items_list:
            return self.__class__(
                data=self.data.limit(0), 
                session=self.session
            )

        return self.__class__(
            data=self.data.filter(F.col(UTILITY_INDEX_COL_NAME).isin(items_list)), 
            session=self.session,
            need_reindex_dense_index=True
        )

    def iloc(self, items: Iterable) -> Iterable:
        raise NotImplementedError("iloc indexing is not supported in Spark backend")


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    @staticmethod
    @contextmanager
    def _cached(df: SparkDF):
        df = df.cache()
        df.count()
        try:
            yield df
        finally:
            df.unpersist()
    
    
    def __init__(self,
                 data: SparkDF | pd.DataFrame | dict[str, Any] | str | None = None,
                 session: SparkSession = None,
                 physical_index_actual_flag = True):
        super().__init__(data = data, session = session, physical_index_actual_flag = physical_index_actual_flag)

    @staticmethod
    def _convert_agg_result(result: SparkDF):
        if len(result.columns) == 1 and result.count() == 1:
            return result.limit(1).take(1)[0][result.columns[0]]
        return result

    def apply(self,
              func: Callable,
              column_name: str | list[str] | None = None,
              axis: int = 0,
              return_type: T.DataType | None = None,
              args: tuple = (),
              kwargs: dict[str, Any] | None = None,
              result_type: Literal["reduce", "expand", "broadcast"] | None = None) -> "SparkDataset":
        kwargs = kwargs or {}
        df = self.data
        
        if axis == 0:
            try:
                test_col = func(F.lit(None))
                is_native = isinstance(test_col, F.Column)
            except Exception:
                is_native = False
            
            if is_native:
                return self._apply_native_func(df, func, column_name)
        
        if return_type is None:
            return_type = self._infer_return_type(df, func, column_name, args, kwargs)
        
        udf_func = F.udf(
            lambda row: func(row.asDict(), *args, **kwargs), 
            return_type
        )
        
        target_cols = self._get_target_columns(df, column_name)
        struct_cols = F.struct(*[F.col(c) for c in target_cols])
        col_expr = udf_func(struct_cols)
        
        new_df = self._apply_udf_result(df, col_expr, return_type, result_type)
        
        return SparkDataset(new_df, self.session)

    def _apply_native_func(self, 
                           df: SparkDF, 
                           func: Callable, 
                           column_name: str | list[str] | None) -> "SparkDataset":
        target_cols = self._get_target_columns(df, column_name)
        
        if column_name is None:
            new_cols = [func(F.col(c)).alias(c) for c in df.columns]
            new_df = df.select(*new_cols)
        else:
            new_cols_dict = {c: func(F.col(c)) for c in target_cols}
            new_df = df.withColumns(new_cols_dict)
        
        return SparkDataset(new_df, self.session)

    def _apply_udf_result(self,
                          df: SparkDF,
                          col_expr: F.Column,
                          return_type: T.DataType,
                          result_type: Literal["reduce", "expand", "broadcast"] | None) -> SparkDF:
        is_struct = isinstance(return_type, T.StructType)
        
        if result_type == "expand" and is_struct:
            return self._expand_struct(df, col_expr, return_type)
        
        elif result_type == "reduce":
            return df.withColumn("result", col_expr)
        
        elif result_type == "broadcast":
            return self._broadcast_result(df, col_expr, return_type)
        
        else:
            if is_struct:
                return self._expand_struct(df, col_expr, return_type)
            else:
                return df.withColumn("result", col_expr)

    def _expand_struct(self, 
                       df: SparkDF, 
                       col_expr: F.Column, 
                       struct_type: T.StructType) -> SparkDF:
        existing_cols = [F.col(c) for c in df.columns]
        new_cols = [F.col(col_expr.alias("__tmp"))[f.name].alias(f.name) 
                    for f in struct_type.fields]
        return df.select(*existing_cols, *new_cols).drop("__tmp")

    def _broadcast_result(self, 
                          df: SparkDF, 
                          col_expr: F.Column, 
                          return_type: T.DataType) -> SparkDF:
        is_struct = isinstance(return_type, T.StructType)
        
        if is_struct:
            temp_df = df.withColumn("__tmp", col_expr)
            new_cols = [F.col("__tmp")[f.name].alias(f.name) 
                        for f in return_type.fields]
            return temp_df.select(*new_cols).drop("__tmp")
        else:
            return df.select([col_expr.alias(c) for c in df.columns])

    def _infer_return_type(self,
                           df: SparkDF,
                           func: Callable,
                           column_name: str | list[str] | None,
                           args: tuple,
                           kwargs: dict[str, Any]) -> T.DataType:
        target_cols = self._get_target_columns(df, column_name)
        
        sample_df = df.select(*target_cols).limit(1)
        if sample_df.isEmpty():
            raise ValueError("Cannot infer return_type: DataFrame is empty")
        
        sample_row = sample_df.collect()[0].asDict()
        
        try:
            sample_res = func(sample_row, *args, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to identify return_type. "
                f"Specify it explicitly: return_type=T.StringType(). "
                f"Error: {str(e)}"
            ) from e
        
        return stm.types(sample_res)

    def _get_target_columns(self, 
                            df: SparkDF, 
                            column_name: str | list[str] | None) -> List[str]:
        if column_name is None:
            return df.columns
        return Adapter.to_list(column_name)

    def map(self,
            func: Callable | dict,
            *,
            return_type: T.DataType | None = None,
            na_action: Literal["ignore"] | None = None,
            **kwargs) -> "SparkDataset":
        df = self.data
        cols = df.columns
        
        if not cols or df.isEmpty():
            return self.__class__(df, self.session)
        
        if isinstance(func, dict):
            if return_type is None:
                sample_val = next(iter(func.values()), None)
                return_type = stm.types(sample_val) if sample_val is not None else T.StringType()
            
            mapping = func
            actual_func = lambda x: mapping.get(x, x)
        else:
            actual_func = func
        
        if return_type is None:
            return_type = self._infer_map_return_type(df, actual_func, cols, kwargs)
        
        udf_func = F.udf(actual_func, return_type)
        
        if na_action == "ignore":
            new_cols = [
                F.when(F.col(c).isNull(), F.col(c))
                .otherwise(udf_func(F.col(c)))
                .alias(c)
                for c in cols
            ]
        elif na_action is None:
            new_cols = [udf_func(F.col(c)).alias(c) for c in cols]
        else:
            raise ValueError(f"Unsupported na_action: {na_action}. Use 'ignore' or None")
        
        new_df = df.select(*new_cols)
        return self.__class__(new_df, self.session)

    def _infer_map_return_type(self,
                               df: SparkDF,
                               func: Callable,
                               cols: list[str],
                               kwargs: dict[str, Any]) -> T.DataType:
        sample_df = df.select(cols[0]).limit(1)
        if sample_df.isEmpty():
            raise ValueError("Cannot infer return_type: DataFrame is empty")
        
        sample_val = sample_df.collect()[0][0]
        
        try:
            result = func(sample_val, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to infer return_type for column '{cols[0]}'. "
                f"Specify explicitly: return_type=T.StringType(). "
                f"Error: {str(e)}"
            ) from e
        
        return stm.types(result)

    def is_empty(self) -> bool:
        return self.data.isEmpty()

    def groupby(self, by: str | Iterable[str], **kwargs) -> list[tuple]:
        if isinstance(by, str):
            by = [by]
        else:
            by = list(by)

        result = []
        with SparkDataset._cached(self.data):
            keys_rows = self.data.select(*by).distinct().orderBy(*by).collect()
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
                result.append((key, SparkDataset(group_df, self.session)))
        
        return result

    def agg(self, func: str | list, **kwargs) -> 'SparkDataset' | float:
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
            if f_name in special_map:
                spark_fn = special_map[f_name]
            else:
                try:
                    spark_fn = getattr(F, f_name)
                except AttributeError:
                    raise ValueError(f"Unsupported agg function: {f_name}")
            
            for c in cols:
                alias = f"{c}_{f_name}" if multi else c
                col_expr = spark_fn(c).alias(alias)
                exprs.append(col_expr)
        
        df_res = df.agg(*exprs)
        res = self._convert_agg_result(df_res)
        
        if isinstance(res, SparkDF):
            return self.__class__(data=res, session=self.session, physical_index_actual_flag=False)
        return res

    def max(self, numeric_only: bool = False) -> 'SparkDataset' | Any:
        return self.agg("max", numeric_only=numeric_only)

    def min(self, numeric_only: bool = False) -> 'SparkDataset' | Any:
        return self.agg("min", numeric_only=numeric_only)

    def count(self, numeric_only: bool = False) -> 'SparkDataset' | int:
        return self.agg("count", numeric_only=numeric_only)

    def sum(self, numeric_only: bool = False) -> 'SparkDataset' | float:
        return self.agg("sum", numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False) -> 'SparkDataset' | float:
        return self.agg("mean", numeric_only=numeric_only)
    
    def var(self, numeric_only: bool = False, ddof: int = 1) -> 'SparkDataset' | float:
        return self.agg("var", numeric_only=numeric_only, ddof=ddof)
    
    def std(self, numeric_only: bool = False, ddof: int = 1) -> 'SparkDataset' | float:
        return self.agg("std", numeric_only=numeric_only, ddof=ddof)

    def mode(self, numeric_only: bool = False, dropna: bool = True) -> 'SparkDataset':
        df = self.data
        if numeric_only:
            cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
        else:
            cols = df.columns
        
        if not cols:
            return self.__class__(df.select([]), self.session)
        
        dfs_to_union = []
        
        with self._cached(df):
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
            return self.__class__(df.select([]), self.session)
        
        final_df = reduce(
            lambda x, y: x.unionByName(y, allowMissingColumns=True), dfs_to_union
        )
        collapse_exprs = [F.first(c, ignorenulls=True).alias(c) for c in cols]
        result_df = (
            final_df.groupBy("row_id")
            .agg(*collapse_exprs)
            .orderBy("row_id")
            .drop("row_id")
        )
        
        return self.__class__(result_df, self.session)


    def log(self) -> 'SparkDataset':
        df = self.data
        numeric_cols = [
            f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)
        ]
        
        if not numeric_cols:
            return self.__class__(df.select([]), self.session)
        
        with self._cached(df):
            exprs = [F.log(F.col(c)).alias(c) for c in numeric_cols]
            result_df = df.select(*exprs)
        
        return self.__class__(result_df, self.session)


    def cov(self, small_format: bool = False) -> 'SparkDataset' | dict:
        col_list = self._public_columns
        n_cols = len(col_list)
        
        if n_cols == 0:
            if small_format:
                return {}
            return self.__class__(self.session.createDataFrame([], StructType([])), self.session)
        
        with self._cached(self.data):
            paired_cov = self.data.select(
                *[
                    F.covar_samp(F.col(col_list[i]), F.col(col_list[j])).alias(f"{i}_{j}")
                    for i in range(n_cols)
                    for j in range(i, n_cols)
                ]
            ).collect()[0]
        
        matrix_data = []
        for i in range(n_cols):
            row = []
            for j in range(n_cols):
                if i <= j:
                    key = f"{i}_{j}"
                else:
                    key = f"{j}_{i}"
                row.append(float(paired_cov[key]) if paired_cov[key] is not None else 0.0)
            matrix_data.append(row)
        
        if small_format:
            return {
                "index": col_list,
                "columns": col_list,
                "data": matrix_data
            }
        else:
            rows = [
                Row(index=col_list[i], **{col_list[j]: matrix_data[i][j] for j in range(n_cols)})
                for i in range(n_cols)
            ]
            
            fields = [StructField("index", StringType(), True)]
            fields += [StructField(c, DoubleType(), True) for c in col_list]
            schema = StructType(fields)
            
            result_df = self.session.createDataFrame(rows, schema)
            return self.__class__(data=result_df, session=self.session, physical_index_actual_flag=False)


    def coefficient_of_variation(self, small_format: bool = False) -> 'SparkDataset' | float | dict:
        col_list = self._public_columns
        n_cols = len(col_list)
        
        if n_cols == 0:
            if small_format:
                return {}
            return self.__class__(self.session.createDataFrame([], StructType([])), self.session)
        
        with self._cached(self.data):
            stats = self.data.select(
                *[
                    F.stddev_samp(F.col(c)).alias(f"{c}_std")
                    for c in col_list
                ] + [
                    F.mean(F.col(c)).alias(f"{c}_mean")
                    for c in col_list
                ]
            ).collect()[0]
            
            cv_values = []
            for c in col_list:
                std_val = stats[f"{c}_std"]
                mean_val = stats[f"{c}_mean"]
                if mean_val is not None and mean_val != 0:
                    cv = float(std_val) / float(mean_val) if std_val is not None else 0.0
                else:
                    cv = 0.0
                cv_values.append(cv)
        
        if n_cols == 1:
            return cv_values[0]
        
        if small_format:
            return {
                "index": ["cv"],
                "columns": col_list,
                "data": [cv_values]
            }
        else:
            row = Row(**{c: cv_values[i] for i, c in enumerate(col_list)})
            result_df = self.session.createDataFrame([row])
            return self.__class__(result_df, self.session)

    def quantile(self, q: float = 0.5) -> float:
        if isinstance(q, (list, tuple)) and len(q) > 1:
            return self.data.approxQuantile(self._public_columns[0], q=q, accuracy=1e-6)
        return self.data.agg(F.expr(f"percentile_approx(`{self._public_columns[0]}`, {q})")).collect()[0][0]

    def get_numeric_columns(self) -> list[str]:
        return [
            field.name
            for field in self.data.schema.fields
            if isinstance(field.dataType, NumericType)
        ]

    def corr(self,
             numeric_only: bool = False,
             small_format: bool = False) -> 'SparkDataset' | pd.DataFrame:
        if numeric_only:
            col_list = self.get_numeric_columns()
        else:
            col_list = self._public_columns
        
        with self._cached(self.data):
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
            return self.__class__(self.session.createDataFrame(result), self.session)

    def isna(self) -> 'SparkDataset':        
        result_df = self.data.select(
            *[
                (
                    (F.isnan(F.col(c)) | F.isnull(F.col(c))).alias(c)
                    if isinstance(self.data.schema[c].dataType, (FloatType, DoubleType))
                    else F.isnull(F.col(c)).alias(c)
                )
                for c in self._public_columns
            ]
        )
        return self.__class__(result_df, self.session)
    
    def na_counts(self) -> 'SparkDataset' | int:
        isna_df = self.isna().data
        result_df = isna_df.agg(*[F.sum(F.col(c).cast("int")).alias(c) for c in isna_df.columns])
        result_row = result_df.collect()[0]
        return {c: int(result_row[c]) if result_row[c] is not None else 0 for c in isna_df.columns}

    def sort_values(self, 
                    by: str | list[str], 
                    ascending: bool = True, 
                    **kwargs) -> 'SparkDataset':
        by = Adapter.to_list(by)
        result_df = self.data.orderBy(*by, ascending=ascending)
        return self.__class__(result_df, self.session)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> 'SparkDataset':
        df = self.data
        cols = df.columns
        
        if not cols:
            return self.__class__(df.select([]), self.session)
        
        with self._cached(df):
            result = df
            if dropna:
                result = result.dropna(how="any")
            
            result = result.groupBy(*cols).count()
            
            if normalize:
                total = result.agg(F.sum("count").alias("total")).collect()[0][0]
                result = result.withColumn("proportion", F.col("count") / F.lit(total))
                result = result.drop("count")
            
            if sort:
                sort_col = "proportion" if normalize else "count"
                result = result.orderBy(F.col(sort_col).desc() if not ascending else F.col(sort_col).asc())
        
        return self.__class__(result, self.session)

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int | None = 0,
    ) -> 'SparkDataset':
        subset = Adapter.to_list(subset) if subset is not None else None
        
        if axis in (0, "index", "rows"):
            result_df = self.data.na.drop(how=how, subset=subset)
            return self.__class__(result_df, self.session)
        
        elif axis in (1, "columns"):
            with self._cached(self.data):
                counts = self.data.select(
                    *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
                    for c in self.columns]
                ).collect()[0].asDict()
                
                if how == "any":
                    keep_cols = [c for c in self.columns if counts[c] == 0]
                else:
                    keep_cols = [c for c in self.columns if counts[c] < self.data.count()]
            
            result_df = self.data.select(*keep_cols)
            return self.__class__(result_df, self.session)
        
        else:
            raise ValueError(f"Invalid axis value: {axis}. Must be 0 or 1.")

    def transpose(self, names: Sequence[str] | None = None) -> 'SparkDataset':
        df = self.data
        df_indexed = df.withColumn("_row_id", F.monotonically_increasing_id())
        original_cols = df.columns
        df_long = df_indexed.select(
            "_row_id",
            F.explode(F.array([
                F.struct(F.lit(c).alias("col_name"), F.col(c).alias("value"))
                for c in original_cols
            ])).alias("col_struct")
        ).select(
            "_row_id",
            F.col("col_struct.col_name").alias("col_name"),
            F.col("col_struct.value").alias("value")
        )
        
        df_transposed = df_long.groupBy("col_name").pivot("_row_id").agg(F.first("value"))
        
        if names is not None:
            for i, new_name in enumerate(names):
                if i < len(df_transposed.columns):
                    df_transposed = df_transposed.withColumnRenamed(
                        df_transposed.columns[i], new_name
                    )
        
        return self.__class__(df_transposed, self.session)

    def sample(
        self,
        frac: float | None = None,
        n: int | None = None,
        random_state: int | None = None,
    ) -> 'SparkDataset':
        if frac is not None and n is not None:
            raise ValueError("Only one of frac or n should be specified")
        if frac is None and n is None:
            raise ValueError("Either frac or n should be specified")
        
        with self._cached(self.data):
            if frac is not None:
                result_df = self.data.sample(
                    withReplacement=False, 
                    fraction=frac, 
                    seed=random_state
                )
                return self.__class__(result_df, self.session)
            
            total = self.data.count()
            if total == 0:
                return self.__class__(self.data, self.session)
            
            fraction = min(float(n) / float(total), 1.0)
            sampled = self.data.sample(
                withReplacement=False, 
                fraction=fraction, 
                seed=random_state
            )
            result_df = sampled.limit(n)
        
        return self.__class__(result_df, self.session)

    def select_dtypes(self,
                      include: str | list[str] | None = None,
                      exclude: str | list[str] | None = None) -> 'SparkDataset':
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
        
        if not dtypes:
            result_df = self.data.select([])
        else:
            result_df = self.data.select([F.col(c).alias(c) for c in dtypes.keys()])
        
        return self.__class__(result_df, self.session)

    def isin(self, values: Iterable) -> 'SparkDataset':
        values = Adapter.to_list(values)
        
        if not values:
            result_df = self.data.select(
                [F.lit(False).alias(c) for c in self._public_columns]
            )
            return self.__class__(result_df, self.session)
        
        col_types = self.get_column_type()
        
        with self._cached(self.data):
            result_df = self.data.select(
                [
                    (
                        F.col(c).isin(values).alias(c)
                        if isinstance(values[0], col_types[c])
                        else F.lit(False).alias(c)
                    )
                    for c in self._public_columns
                ]
            )
        
        return self.__class__(result_df, self.session)

    def merge(
        self,
        right: 'SparkDataset',
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool | None = None,
        right_index: bool | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner") -> 'SparkDataset':
        left_df = self.data
        right_df = right.data
        
        if left_index or right_index:
            raise NotImplementedError(
                "Merging by index is not supported in Spark. "
                "Use 'on', 'left_on', or 'right_on' parameters instead."
            )
        
        if on is not None:
            left_on = on
            right_on = on
        
        if left_on is None or right_on is None:
            raise MergeOnError(
                f"Must specify 'on' or both 'left_on' and 'right_on'. "
                f"Got: on={on}, left_on={left_on}, right_on={right_on}"
            )
        
        with self._cached(left_df):
            with self._cached(right_df):
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
                
                if "__index__" in result.columns:
                    result = result.drop("__index__")
        
        return self.__class__(data=result, 
                              session=self.session, 
                              physical_index_actual_flag=False)

    def drop(self, 
             labels: Any = "", 
             axis: int = 1) -> 'SparkDataset':
        labels = Adapter.to_list(labels)
        
        if axis == 1:
            result_df = self.data.drop(*labels)
            return self.__class__(result_df, self.session)
        elif axis == 0:
            raise NotImplementedError(
                "Dropping rows by index is not supported in Spark. "
                "Use SQL conditions instead (e.g., ds.data.filter(F.col('age') > 18))"
            )
        else:
            raise ValueError(f"Invalid axis value: {axis}. Must be 0 or 1.")

    def filter(self,
               items: list | None = None,
               regex: str | None = None,
               axis: int = 1) -> 'SparkDataset':
        if axis == 1:
            if items is None and regex is not None:
                items = [col for col in self._public_columns if re.search(regex, col)]
            
            if items is None:
                return self.__class__(self.data.select([]), self.session)
            
            result_df = self.data.select(items)
            return self.__class__(result_df, self.session)
        
        elif axis == 0:
            raise NotImplementedError(
                "Filtering rows by index list is not supported in Spark. "
                "Use SQL conditions instead (e.g., ds.filter(F.col('age') > 18))"
            )
        else:
            raise ValueError(f"Invalid axis value: {axis}. Must be 0 or 1.")

    def rename(self, columns: dict[str, str]) -> 'SparkDataset':
        df = self.data
        for old_name, new_name in columns.items():
            df = df.withColumnRenamed(old_name, new_name)
        return self.__class__(df, self.session)

    def replace(self,
                to_replace: Any = None,
                value: Any = None,
                subset: list[str] | None = None,
                regex: bool = False) -> 'SparkDataset':
        if regex:
            if isinstance(to_replace, str) and isinstance(value, str):
                cols = subset if subset else self._public_columns
                df = self.data
                for col_name in cols:
                    df = df.withColumn(col_name, F.regexp_replace(col_name, to_replace, value))
                return self.__class__(df, self.session)
            else:
                raise NotImplementedError("Regex replacement with lists/dicts is not supported")

        mapping_dict: Dict[Any, Any] = {}
        
        if isinstance(to_replace, dict):
            mapping_dict = to_replace
        elif isinstance(to_replace, list):
            if isinstance(value, list):
                if len(to_replace) != len(value):
                    raise ValueError("Replacement lists must be of the same length")
                mapping_dict = dict(zip(to_replace, value))
            else:
                mapping_dict = {k: value for k in to_replace}
        else:
            mapping_dict = {to_replace: value}

        if not mapping_dict:
            return self

        all_cols = self._public_columns
        target_cols_set = set(subset) if subset else set(all_cols)
        
        col_types = {f.name: f.dataType for f in self.data.schema.fields}
        
        result_cols = []
        
        for col_name in all_cols:
            if col_name not in target_cols_set:
                result_cols.append(F.col(col_name))
                continue
            
            col_type = col_types[col_name]
            col_expr = F.col(col_name)
            replacement_applied = False
            
            for old_val, new_val in mapping_dict.items():
                type_compatible = False
                
                if old_val is None:
                    type_compatible = True
                elif isinstance(col_type, StringType):
                    type_compatible = isinstance(old_val, str)
                elif isinstance(col_type, BooleanType):
                    type_compatible = isinstance(old_val, bool)
                elif isinstance(col_type, (IntegerType, LongType, ShortType, ByteType)):
                    type_compatible = isinstance(old_val, int) and not isinstance(old_val, bool)
                elif isinstance(col_type, (FloatType, DoubleType, DecimalType)):
                    type_compatible = isinstance(old_val, (int, float)) and not isinstance(old_val, bool)
                elif isinstance(col_type, (DateType, TimestampType)):
                    type_compatible = isinstance(old_val, (str,)) 
                else:
                    type_compatible = False
                
                if type_compatible:
                    try:
                        if old_val is None:
                            condition = col_expr.isNull()
                        else:
                            condition = (col_expr == F.lit(old_val))
                        
                        col_expr = F.when(condition, F.lit(new_val)).otherwise(col_expr)
                        replacement_applied = True
                    except Exception as e:
                        continue
            
            result_cols.append(col_expr.alias(col_name))

        result_df = self.data.select(*result_cols)
        return self.__class__(result_df, self.session)

    def reindex(self, 
                labels: str | list[str] = "", 
                fill_value: str | None = None) -> 'SparkDataset':
        labels = Adapter.to_list(labels)
        existing = self._public_columns
        selected = []
        
        for c in labels:
            if c in existing:
                selected.append(F.col(c).alias(c))
            else:
                selected.append(F.lit(fill_value).alias(c))
        
        result_df = self.data.select(*selected)
        return self.__class__(result_df, self.session)

    def list_to_columns(self, column: str) -> 'SparkDataset':
        df = self.data
        
        with self._cached(df):
            first_row = df.select(column).head(1)
            if first_row is None or first_row[0][0] is None:
                n_cols = 0
            else:
                n_cols = len(first_row[0][0])
            
            if n_cols == 0:
                return self.__class__(df.select([]), self.session)
            
            exprs = [F.col(column)[i].alias(f"{column}_{i}") for i in range(n_cols)]
            result_df = df.select(*exprs)
        
        return self.__class__(result_df, self.session)
        
    def fillna(self,
               values: ScalarType | dict[str, ScalarType] | None = None,
               method: Literal["bfill", "ffill"] | None = None,
               **kwargs) -> 'SparkDataset':
        if method is not None:
            raise NotImplementedError(
                "Fill method ('ffill', 'bfill') requires row ordering (Window functions). "
                "In Spark backend, this requires an explicit ordering column. "
                "Please use SQL conditions or add an index column before calling fillna with method."
            )
        
        if values is None:
            return self
        
        try:
            new_df = self.data.na.fill(values)
            return self.__class__(new_df, self.session)
        except Exception as e:
            raise TypeError(f"Failed to fill NA values: {str(e)}")

    def dot(self,
            other: SparkDataset | np.ndarray,
            broadcast_threshold_mb: float = 100.0,
            auto_broadcast: bool = True,
            result_col_prefix: str = "") -> 'SparkDataset':
        raise NotImplementedError("dot is not supported in Spark backend")
        
    def idxmax(self):
        raise NotImplementedError("idxmax is not supported in Spark backend")
        
    def sort_index(self, ascending: bool = True, **kwargs) -> 'SparkDataset':
        raise NotImplementedError("sort_index is not supported in Spark backend")
    
    def unique(self) -> 'SparkDataset':
        raise NotImplementedError("unique is not supported in Spark backend")

    def nunique(self, dropna: bool = True) -> 'SparkDataset':
        raise NotImplementedError("nunique is not supported in Spark backend")