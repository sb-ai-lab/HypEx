from __future__ import annotations

import os
import sys
import re
from itertools import chain, reduce

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
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame as SparkDF

from pyspark.sql.types import (
    NumericType,
    ArrayType,
    StructType,
    StructField,
    StringType,
    DoubleType,
)

from ...utils import FromDictTypes, MergeOnError, ScalarType, Adapter
from ...utils.typings import SparkTypeMapper as stm
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(filename: str | Path, session: SparkSession) -> spark.DataFrame:
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
                 data: spark.DataFrame | pd.DataFrame | dict[str, Any] | str | None = None,
                 session: SparkSession | None = None):
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

        if isinstance(data, spark.DataFrame):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.data = self.session.createDataFrame(data)
        elif isinstance(data, str):
            self.data = self._read_file(data, self.session)
        else:
            self.data = self.session.createDataFrame([], schema=StructType([]))
        self.count_data: int | None = None
            
    def __add_row_index(self, df: spark.DataFrame, index_column_name: str) -> spark.DataFrame:
        """Not used."""
        rdd = df.rdd.zipWithIndex()
        new_rdd = rdd.map(lambda x: tuple(x[0]) + (x[1],))
        new_schema = T.StructType(df.schema.fields + [T.StructField(index_column_name, T.LongType(), False)])
        return self.session.createDataFrame(new_rdd, schema=new_schema)

    def __getitem__(self, item: Any):
        raise NotImplementedError("Spark-base Dataset does not support indexing")

    def __len__(self) -> int:
        if self.count_data is None:
            self.count_data = 0 if self.data is None else self.data.count()
        return self.count_data

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
        
    def add_column(self, data: spark.DataFrame | list, name: str | None = None, index = None) -> 'SparkNavigation':
        raise NotImplementedError("For add column need implement uniq_column")

    # comparison operators:
    def __eq__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) == other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __ne__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) != other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __lt__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) < other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __le__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) <= other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __ge__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) >= other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __gt__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) > other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    # unary operators:
    def __pos__(self) -> 'SparkNavigation':
        result_df = self.data.select([F.col(c).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __neg__(self) -> 'SparkNavigation':
        result_df = self.data.select([(-F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __abs__(self) -> 'SparkNavigation':
        result_df = self.data.select([F.abs(F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __invert__(self) -> 'SparkNavigation':
        result_df = self.data.select([(~F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __round__(self, n: int = 0) -> 'SparkNavigation':
        result_df = self.data.select([F.round(F.col(c), n).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    # binary arithmetic operations:
    def __add__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) + other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __sub__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) - other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __mul__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) * other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __floordiv__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.floor(F.col(c) / other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __truediv__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) / other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __mod__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) % other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __pow__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.pow(F.col(c), other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    # binary logical operators:
    def __and__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) & other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __or__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(F.col(c) | other_val).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __xor__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        try:
            result_df = self.data.select([F.xor(F.col(c), other_val).alias(c) for c in self.data.columns])
        except (AttributeError, TypeError):
            result_df = self.data.select([
                ((F.col(c) & ~other_val) | (~F.col(c) & other_val)).alias(c) 
                for c in self.data.columns
            ])
        return SparkNavigation(data=result_df, session=self.session)

    # reverse arithmetic operators:
    def __radd__(self, other) -> 'SparkNavigation':
        return self.__add__(other)

    def __rsub__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(other_val - F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __rmul__(self, other) -> 'SparkNavigation':
        return self.__mul__(other)

    def __rfloordiv__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.floor(other_val / F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __rtruediv__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(other_val / F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __rdiv__(self, other) -> 'SparkNavigation':
        return self.__rtruediv__(other)

    def __rmod__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([(other_val % F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    def __rpow__(self, other) -> 'SparkNavigation':
        other_val = self.__magic_determine_other(other)
        result_df = self.data.select([F.pow(other_val, F.col(c)).alias(c) for c in self.data.columns])
        return SparkNavigation(data=result_df, session=self.session)

    # representation methods:
    def __repr__(self) -> str:
        if self.data is None:
            return "SparkNavigation(data=None)"
        
        rows = self.data.count()
        columns = len(self.data.columns)
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
            columns = len(self.data.columns)
            
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
            existing_cols = [k for k in key_list if k in self.data.columns]
            if existing_cols:
                return SparkNavigation(data=self.data.select(*existing_cols), session=self.session)
            else:
                return default
        elif isinstance(key_list[0], int):
            raise NotImplementedError("Access by row index list is not supported")
        else:
            return default

    def take(self, indices: int | list[int], axis: Literal["index", "columns", "rows"] | int = 0) -> 'SparkNavigation':
        if not isinstance(indices, list):
            indices = [indices]

        if axis in (1, "columns"):
            col_names = []
            for idx in indices:
                if 0 <= idx < len(self.data.columns):
                    col_names.append(self.data.columns[idx])
                else:
                    raise IndexError(f"Column index {idx} out of range")
            
            return SparkNavigation(data=self.data.select(*col_names), session=self.session)

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
        
        return SparkNavigation(data=new_df, session=self.session)

    @property
    def columns(self) -> list[str]:
        return self.data.columns if self.data else []

    @property
    def shape(self) -> tuple[int, int]:
        if self.data:
            count = self.data.count()
            cols = len(self.data.columns)
            return (count, cols)
        return (0, 0)

    def get_column_index(self, column_name: Sequence[str] | str) -> int | Sequence[int]:
        pd_index_columns = pd.Index(self.data.columns)
        if isinstance(column_name, str):
            return pd_index_columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return pd_index_columns.get_indexer(column_name)
        else:
            raise TypeError("Wrong column_name type.")

    def get_column_type(self, column_name: list[str] | str = None) -> dict[str, type] | type | None:
        if column_name is None:
            column_name = self.data.columns
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

    def astype(self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise") -> 'SparkNavigation':
        for col, new_type in dtype.items():
            if col not in self.data.columns:
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
        
        return SparkNavigation(data=new_df, session=self.session)

    def update_column_type(self, dtype: dict[str, type]):
        if len(dtype) > 0:
            self.data = self.astype(dtype)
        return self

    def append(self, 
               other: 'SparkNavigation | list[SparkNavigation]',
               axis: int = 0) -> 'SparkNavigation':
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
            return SparkNavigation(data=new_data, session=self.session)
        
        if axis == 1:
            raise NotImplementedError("append on axis == 1 is not supported in Spark backend")
        
        raise ValueError("axis should be 0 or 1")
    
    @property
    def index(self):
        raise AttributeError("Spark-based Dataset has no index")

    def from_dict(self, data: FromDictTypes, index: Optional[Union[Iterable, Sized]] = None):
        raise NotImplementedError("Need implement")

    def get_values(self, row: str | int | None = None, column: str | None = None) -> Any:
        raise NotImplementedError("Row-based value access is not supported")

    def iget_values(self, row: int | None = None, column: int | None = None) -> Any:
        raise NotImplementedError("Row-based value access is not supported")

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("to_dict is not supported in Spark backend")

    def to_records(self) -> list[dict]:
        raise NotImplementedError("to_records is not supported in Spark backend")

    def loc(self, items: Iterable) -> Iterable:
        raise NotImplementedError("loc indexing is not supported in Spark backend")


    def iloc(self, items: Iterable) -> Iterable:
        raise NotImplementedError("iloc indexing is not supported in Spark backend")


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    def __init__(self,
                 data: spark.DataFrame | pd.DataFrame | dict | str | None = None,
                 session: SparkSession = None):
        super().__init__(data, session)

    def __deepcopy__(self, memo):
        return SparkDataset(data=self.data.select("*"), session=self.session)

    @staticmethod
    def _convert_agg_result(result: spark.DataFrame):
        if len(result.columns) == 1 and result.count() == 1:
            return result.limit(1).take(1)[0][result.columns[0]]
        return result

    def apply(self,
              func: Callable,
              column_name: str | list[str] | None = None,
              axis: int = 0,
              return_type: Optional[T.DataType] = None,
              args: tuple = (),
              kwargs: Optional[Dict[str, Any]] = None,
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
        
        return self.__class__(new_df, self.session)

    def _apply_native_func(self, 
                           df: spark.DataFrame, 
                           func: Callable, 
                           column_name: str | list[str] | None) -> "SparkDataset":
        target_cols = self._get_target_columns(df, column_name)
        
        if column_name is None:
            new_cols = [func(F.col(c)).alias(c) for c in df.columns]
            new_df = df.select(*new_cols)
        else:
            new_cols_dict = {c: func(F.col(c)) for c in target_cols}
            new_df = df.withColumns(new_cols_dict)
        
        return self.__class__(new_df, self.session)

    def _apply_udf_result(self,
                          df: spark.DataFrame,
                          col_expr: F.Column,
                          return_type: T.DataType,
                          result_type: Literal["reduce", "expand", "broadcast"] | None) -> spark.DataFrame:
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
                       df: spark.DataFrame, 
                       col_expr: F.Column, 
                       struct_type: T.StructType) -> spark.DataFrame:
        existing_cols = [F.col(c) for c in df.columns]
        new_cols = [F.col(col_expr.alias("__tmp"))[f.name].alias(f.name) 
                    for f in struct_type.fields]
        return df.select(*existing_cols, *new_cols).drop("__tmp")

    def _broadcast_result(self, 
                          df: spark.DataFrame, 
                          col_expr: F.Column, 
                          return_type: T.DataType) -> spark.DataFrame:
        is_struct = isinstance(return_type, T.StructType)
        
        if is_struct:
            temp_df = df.withColumn("__tmp", col_expr)
            new_cols = [F.col("__tmp")[f.name].alias(f.name) 
                        for f in return_type.fields]
            return temp_df.select(*new_cols).drop("__tmp")
        else:
            return df.select([col_expr.alias(c) for c in df.columns])

    def _infer_return_type(self,
                           df: spark.DataFrame,
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
        
        return d.types(sample_res)

    def _get_target_columns(self, 
                            df: spark.DataFrame, 
                            column_name: str | list[str] | None) -> List[str]:
        if column_name is None:
            return df.columns
        return Adapter.to_list(column_name)

    def map(self,
            func: Callable | dict,
            *,
            return_type: T.DataType | None = None,
            na_action: Optional[Literal["ignore"]] = None,
            **kwargs) -> "SparkDataset":
        df = self.data
        cols = df.columns
        
        if not cols or df.isEmpty():
            return self.__class__(df, self.session)
        
        if isinstance(func, dict):
            if return_type is None:
                sample_val = next(iter(func.values()), None)
                return_type = d.types(sample_val) if sample_val is not None else T.StringType()
            
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

    def _infer_map_return_type(
        self,
        df: spark.DataFrame,
        func: Callable,
        cols: List[str],
        kwargs: Dict[str, Any]
    ) -> T.DataType:
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
        
        return d.types(result)

    def is_empty(self) -> bool:
        return self.data.isEmpty()


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

    def agg(self, func: str | list, **kwargs) -> Union[spark.DataFrame, float]:
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
        if isinstance(q, (list, tuple)) and len(q) > 1:
            return self.data.approxQuantile(self.data.columns[0], q=q, accuracy=1e-6)
        return self.data.agg(F.expr(f"percentile_approx(`{self.data.columns[0]}`, {q})")).collect()[0][0]

    def coefficient_of_variation(
        self, small_format=False
    ) -> Union[spark.DataFrame, float]:
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

    def get_numeric_columns(self) -> list[str]:
        return [
            field.name
            for field in self.data.schema.fields
            if isinstance(field.dataType, NumericType)
        ]

    def corr(self,
             numeric_only: bool = False,
             small_format: bool = False) -> Union[spark.DataFrame, float]:

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

    def sort_values(self, by: str | list[str], ascending: bool = True, **kwargs) -> spark.DataFrame:
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
        values: Optional[Union[ScalarType, dict[str, ScalarType]]] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
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

    def na_counts(self) -> Union[spark.DataFrame, int]:
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
        df2: Union[spark.DataFrame, np.ndarray],
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
        subset: Optional[Union[str, Iterable[str]]] = None,
        axis: Union[Literal["index", "rows", "columns"], int] = 0,
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
        self, names: Optional[Sequence[str]] = None
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
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        left_index: Optional[bool] = None,
        right_index: Optional[bool] = None,
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
            raise NotImplementedError("Dropping rows by index is not supported in Spark")

        else:
            raise ValueError("Invalid axis value")

    def filter(
        self,
        items: Optional[list] = None,
        regex: Optional[str] = None,
        axis: int = 0,
    ) -> spark.DataFrame:
        if axis == 1:
            if items is None and regex is not None:
                items = [col for col in self.data.columns if re.search(regex, col)]
            return self.data.select(items)
        elif axis == 0:
            raise NotImplementedError("Filtering rows by index list is not supported. Use SQL conditions instead.")

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
        self, labels: str = "", fill_value: Optional[str] = None
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
        
    def idxmax(self):
        raise NotImplementedError("idxmax is not supported in Spark backend")
        
    def sort_index(self, ascending: bool = True, **kwargs) -> spark.DataFrame:
        raise NotImplementedError("sort_index is not supported in Spark backend")
    
    def unique(self) -> SparkDF:
        raise NotImplementedError("unique is not supported in Spark backend")

    def nunique(self, dropna: bool = True) -> SparkDF:
        raise NotImplementedError("nunique is not supported in Spark backend")