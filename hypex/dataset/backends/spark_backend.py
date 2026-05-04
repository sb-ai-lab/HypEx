from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, Sized

import numpy as np
import pandas as pd

from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDF
import pyspark.sql.functions as F
from pyspark.sql.types import StructType


import pyspark.pandas as ps
ps.set_option('compute.ops_on_diff_frames', True)

from pyspark.pandas.exceptions import PandasNotImplementedError


        
from ...utils import FromDictTypes, MergeOnError, ScalarType, SparkTypeMapper
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    """Navigation interface for PySpark-backed datasets.
    
    Provides pandas-like indexing, slicing, and basic operations on distributed
    DataFrames using pyspark.pandas as the backend. Handles automatic conversion
    between Spark DataFrames and pyspark.pandas DataFrames, with safeguards for
    large data conversions to local pandas.
    
    Attributes:
        PANDAS_CONVERSION_LIMIT (int): Maximum number of rows allowed when 
            converting to local pandas to prevent memory issues.
        data (ps.DataFrame): The underlying pyspark.pandas DataFrame.
        session (SparkSession): Active Spark session for distributed operations.
    """
    PANDAS_CONVERSION_LIMIT: int = 100_000

    def checkpoint(self):
        """Create a checkpoint in the Spark execution plan.
        
        Raises:
            NotImplementedError: This method is not implemented for SparkNavigation.
                Use Spark-specific checkpointing mechanisms instead.
        """
        raise NotImplementedError("Method checkpoint not implemented for SparkNavigation.")
        
    def limit(self, num: int | None = None) -> Any:
        """Limit the number of rows in the dataset.
        
        Args:
            num (int | None): Maximum number of rows to return. If None, 
                returns all rows.
                
        Returns:
            Any: Wrapped result containing limited data, preserving the 
                SparkNavigation interface for chaining.
        """
        return self._wrap_result(self.data.iloc[:num])
    
    def _check_pandas_conversion(self, obj: ps.DataFrame | ps.Series, context: str = "") -> None:
        """Validate that converting to pandas won't exceed memory limits.
        
        Args:
            obj (ps.DataFrame | ps.Series): The pyspark.pandas object to check.
            context (str): Optional context string for error messages.
            
        Raises:
            ValueError: If the object contains more rows than PANDAS_CONVERSION_LIMIT.
        """
        n: int = obj.__len__()
        if n > self.PANDAS_CONVERSION_LIMIT:
            raise ValueError(f"{context}: {n} rows exceed limit {self.PANDAS_CONVERSION_LIMIT}")
        
    def _wrap_result(self, 
                     result: ps.DataFrame | ps.Series | Any) -> "SparkNavigation" | ps.Series | Any:
        """Wrap operation results to maintain consistent interface.
        
        Converts raw pyspark.pandas results back into SparkNavigation instances
        for method chaining, or returns scalar values directly.
        
        Args:
            result (ps.DataFrame | ps.Series | Any): Result from a DataFrame operation.
            
        Returns:
            SparkNavigation | ps.Series | Any: Wrapped result preserving the 
                appropriate type for further operations.
        """
        if isinstance(result, ps.DataFrame):
            return self.__class__(data=result, session=self.session)
        
        if isinstance(result, ps.Series):
            return self.__class__(data=result.to_frame(), session=self.session)

        return result

    @staticmethod
    def _read_file(filename: str | Path, session: SparkSession) -> ps.DataFrame:
        """Read a file into a pyspark.pandas DataFrame.
        
        Supports multiple file formats with automatic format detection based on
        file extension. Handles permissions and path validation.
        
        Args:
            filename (str | Path): Path to the file to read.
            session (SparkSession): Active Spark session for reading distributed data.
            
        Returns:
            ps.DataFrame: Loaded data as a pyspark.pandas DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a file or has unsupported extension.
            PermissionError: If the file cannot be read.
        """
        file_path = Path(filename).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: '{file_path}'")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: '{file_path}'")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied: '{file_path}'")

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            spark_df = (
                session.read.format("csv")
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
            raise ValueError(
                f"Unsupported file extension: '{suffix}'. "
                f"Supported: .csv, .parquet, .json, .orc, .xlsx"
            )

        return ps.DataFrame(spark_df)

    def __init__(self,
                 data: ps.DataFrame | pd.DataFrame | SparkDF | dict[str, Any] | str | None = None,
                 session: SparkSession | None = None):
        """Initialize a SparkNavigation instance.
        
        Accepts various input types and normalizes them to pyspark.pandas DataFrame
        format. Automatically infers SparkSession from input data when possible.
        
        Args:
            data (ps.DataFrame | pd.DataFrame | SparkDF | dict[str, Any] | str | None): 
                Source data in various formats:
                - pyspark.pandas/pandas DataFrame or Spark DataFrame
                - Dictionary with 'data' and optional 'index' keys
                - File path string (auto-detected format)
                - None for empty DataFrame
            session (SparkSession | None): Spark session for distributed operations.
                Required if not inferable from data.
                
        Raises:
            ValueError: If session cannot be inferred and is not provided.
            TypeError: If session is not a SparkSession or data type is unsupported.
        """
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
            self.data = ps.DataFrame(
                self.session.createDataFrame([], schema=StructType([]))
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
    def persist(self, 
                storage_level: Literal["MEMORY_ONLY", 
                                       "MEMORY_AND_DISK", 
                                       "DISK_ONLY"] = "MEMORY_AND_DISK",
                action: Literal["count", "head", "none"] = "count"):
        """Persist the underlying Spark DataFrame in cache with automatic materialization.
        
        Marks the dataset for caching in Spark's execution engine to accelerate
        subsequent operations. Unlike Spark's native `persist()`, this method
        optionally triggers an action to immediately materialize the cache,
        ensuring the data is pre-computed and ready for fast access.
        
        Handles index preservation across pyspark.pandas ↔ Spark DataFrame conversions.
        
        Args:
            storage_level (Literal): Storage strategy for cached 
                - "MEMORY_ONLY": Store as deserialized Java objects in heap memory.
                - "MEMORY_AND_DISK": Store in memory, spill partitions to disk if needed.
                - "DISK_ONLY": Store partitions only on disk.
                Default is "MEMORY_AND_DISK" for balanced performance/reliability.
                
            action (Literal): Action to trigger cache materialization:
                - "count": Execute `count()` action (fast, returns row count).
                - "head": Execute `head(1)` action (materializes first partition).
                - "none": Skip automatic action; cache will materialize on first action.
                Default is "count" for reliable full materialization.
                
        Returns:
            Self: Reference to self for method chaining.
            
        Raises:
            ValueError: If storage_level or action parameter has invalid value.
            
        Note:
            - Index is preserved across the Spark conversion cycle.
            - MultiIndex is supported: all index levels are restored after persist.
            - Cached data remains until `unpersist()` is called or SparkSession ends.
            - Use `unpersist()` to manually release cached resources.
        """            
        storage_levels = {
            "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
            "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
            "DISK_ONLY": StorageLevel.DISK_ONLY,
        }
        
        if storage_level not in storage_levels:
            raise ValueError(
                f"Invalid storage_level: '{storage_level}'. "
                f"Valid options: {list(storage_levels.keys())}"
            )
        
        if action not in ("count", "head", "none"):
            raise ValueError(
                f"Invalid action: '{action}'. Valid options: 'count', 'head', 'none'"
            )
        
        original_index_names = self.data.index.names
        original_index_name = original_index_names[0] if len(original_index_names) == 1 else original_index_names
        
        self.data = self.data.spark.persist(storage_levels[storage_level])
        
        if action == "count":
            _ = self.data.count()
        elif action == "head":
            _ = self.data.head(1)
        
        if isinstance(original_index_name, str):
            if original_index_name in self.data.columns:
                self.data = self.data.set_index(original_index_name)
        elif isinstance(original_index_name, list):
            if all(name in self.data.columns for name in original_index_name):
                self.data = self.data.set_index(original_index_name)
        elif original_index_names == [None] and "index" in self.data.columns:
            self.data = self.data.set_index("index")
            
        self._is_persisted_flag = True
        self._storage_level_flag = storage_level
        
        return self
    
    def unpersist(self, blocking: bool = False):
        """Remove the persisted dataset from Spark cache.
        
        Releases memory/disk resources occupied by cached data. After calling
        this method, subsequent operations will recompute data from source.
        
        Args:
            blocking (bool): If True, wait until all blocks are deleted 
                before returning. If False (default), deletion happens 
                asynchronously.
                
        Returns:
            Self: Reference to self for method chaining.
        """
        if getattr(self, '_is_persisted_flag', False):
            original_index_names = self.data.index.names
            original_index_name = original_index_names[0] if len(original_index_names) == 1 else original_index_names
            
            if blocking:
                self.data.to_spark().unpersist(blocking=True)
            else:
                self.data.spark.unpersist()
            
            if isinstance(original_index_name, str):
                if original_index_name in self.data.columns:
                    self.data = self.data.set_index(original_index_name)
            elif isinstance(original_index_name, list):
                if all(name in self.data.columns for name in original_index_name):
                    self.data = self.data.set_index(original_index_name)
            elif original_index_names == [None] and "index" in self.data.columns:
                self.data = self.data.set_index("index")
            
            self._is_persisted_flag = False
            self._storage_level_flag = None
        
        return self
    
    @property
    def is_persisted(self) -> bool:
        """Check if the underlying Spark DataFrame is persisted in cache.
        
        Returns:
            bool: True if DataFrame has non-NONE storage level.
        """        
        return getattr(self, '_is_persisted_flag', False)

    def get_storage_level(self) -> str | None:
        """Get the current storage level of the Spark DataFrame.
        
        Returns:
            str | None: Storage level name (e.g., "MEMORY_AND_DISK") if persisted,
                None if not persisted.
        """        
        return getattr(self, '_storage_level_flag', None)

    def __getitem__(self, 
                    item: slice | int | str | list | ps.DataFrame | ps.Series) -> "SparkNavigation" | ps.Series:
        """Support indexing and column selection operations.
        
        Handles multiple indexing patterns:
        - Integer/slice: row selection via iloc
        - String: column selection by name
        - List: multiple column selection
        - DataFrame/Series: boolean masking
        
        Args:
            item (slice | int | str | list | ps.DataFrame | ps.Series): 
                Indexing specification.
                
        Returns:
            SparkNavigation | ps.Series: Selected data, wrapped appropriately.
            
        Raises:
            ValueError: If boolean DataFrame mask has multiple columns.
            KeyError: If column or row specification is invalid.
        """        
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
        """Return the number of rows in the dataset.
        
        Returns:
            int: Row count of the underlying DataFrame.
        """
        return len(self.data)

    @staticmethod
    def __magic_determine_other(other: Any) -> Any:
        """Extract underlying data for binary operations.
        
        Helper method to handle operations between SparkNavigation instances
        and other types by extracting the raw DataFrame when needed.
        
        Args:
            other (Any): Operand in a binary operation.
            
        Returns:
            Any: The underlying data attribute if other is SparkDataset, 
                otherwise the original value.
        """
        if isinstance(other, SparkDataset):
            return other.data
        else:
            return other

    # comparison operators:
    def __eq__(self, other: Any) -> "SparkNavigation":
        """Element-wise equality comparison."""
        return self._wrap_result(self.data == self.__magic_determine_other(other))

    def __ne__(self, other: Any) -> "SparkNavigation":
        """Element-wise inequality comparison."""
        return self._wrap_result(self.data != self.__magic_determine_other(other))

    def __le__(self, other: Any) -> "SparkNavigation":
        """Element-wise less-than-or-equal comparison."""
        return self._wrap_result(self.data <= self.__magic_determine_other(other))

    def __lt__(self, other: Any) -> "SparkNavigation":
        """Element-wise less-than comparison."""
        return self._wrap_result(self.data < self.__magic_determine_other(other))

    def __ge__(self, other: Any) -> "SparkNavigation":
        """Element-wise greater-than-or-equal comparison."""
        return self._wrap_result(self.data >= self.__magic_determine_other(other))

    def __gt__(self, other: Any) -> "SparkNavigation":
        """Element-wise greater-than comparison."""
        return self._wrap_result(self.data > self.__magic_determine_other(other))

    # unary operations:
    def __pos__(self) -> "SparkNavigation":
        """Unary positive operation (no-op for numeric data)."""
        return self._wrap_result(+self.data)

    def __neg__(self) -> "SparkNavigation":
        """Unary negation operation."""
        return self._wrap_result(-self.data)

    def __abs__(self) -> "SparkNavigation":
        """Element-wise absolute value."""
        return self._wrap_result(abs(self.data))

    def __invert__(self) -> "SparkNavigation":
        """Element-wise logical NOT for boolean data."""
        return self._wrap_result(~self.data)

    def __round__(self, ndigits: int = 0) -> "SparkNavigation":
        """Round numeric values to specified decimal places.
        
        Args:
            ndigits (int): Number of decimal places for rounding.
        """
        return self._wrap_result(self.data.round(ndigits))

    # Binary operations:
    def __add__(self, other: Any) -> "SparkNavigation":
        """Element-wise addition."""
        return self._wrap_result(self.data + self.__magic_determine_other(other))

    def __sub__(self, other: Any) -> "SparkNavigation":
        """Element-wise subtraction."""
        return self._wrap_result(self.data - self.__magic_determine_other(other))

    def __mul__(self, other: Any) -> "SparkNavigation":
        """Element-wise multiplication."""
        return self._wrap_result(self.data * self.__magic_determine_other(other))

    def __floordiv__(self, other: Any) -> "SparkNavigation":
        """Element-wise floor division."""
        return self._wrap_result(self.data // self.__magic_determine_other(other))

    def __div__(self, other: Any) -> "SparkNavigation":
        """Element-wise division (legacy operator)."""
        return self._wrap_result(self.data / self.__magic_determine_other(other))

    def __truediv__(self, other: Any) -> "SparkNavigation":
        """Element-wise true division."""
        return self._wrap_result(self.data / self.__magic_determine_other(other))

    def __mod__(self, other: Any) -> "SparkNavigation":
        """Element-wise modulo operation."""
        return self._wrap_result(self.data % self.__magic_determine_other(other))

    def __pow__(self, other: Any) -> "SparkNavigation":
        """Element-wise exponentiation."""
        return self._wrap_result(self.data ** self.__magic_determine_other(other))

    def __and__(self, other: Any) -> "SparkNavigation":
        """Element-wise logical AND for boolean data."""
        return self._wrap_result(self.data & self.__magic_determine_other(other))

    def __or__(self, other: Any) -> "SparkNavigation":
        """Element-wise logical OR for boolean data."""
        return self._wrap_result(self.data | self.__magic_determine_other(other))

    # Right arithmetic operators:
    def __radd__(self, other: Any) -> "SparkNavigation":
        """Right-hand addition (other + self)."""
        return self._wrap_result(self.__magic_determine_other(other) + self.data)

    def __rsub__(self, other: Any) -> "SparkNavigation":
        """Right-hand subtraction (other - self)."""
        return self._wrap_result(self.__magic_determine_other(other) - self.data)

    def __rmul__(self, other: Any) -> "SparkNavigation":
        """Right-hand multiplication (other * self)."""
        return self._wrap_result(self.__magic_determine_other(other) * self.data)

    def __rfloordiv__(self, other: Any) -> "SparkNavigation":
        """Right-hand floor division (other // self)."""
        return self._wrap_result(self.__magic_determine_other(other) // self.data)

    def __rdiv__(self, other: Any) -> "SparkNavigation":
        """Right-hand division (legacy operator, other / self)."""
        return self._wrap_result(self.__magic_determine_other(other) / self.data)

    def __rtruediv__(self, other: Any) -> "SparkNavigation":
        """Right-hand true division (other / self)."""
        return self._wrap_result(self.__magic_determine_other(other) / self.data)

    def __rmod__(self, other: Any) -> "SparkNavigation":
        """Right-hand modulo (other % self)."""
        return self._wrap_result(self.__magic_determine_other(other) % self.data)

    def __rpow__(self, other: Any) -> "SparkNavigation":
        """Right-hand exponentiation (other ** self)."""
        return self._wrap_result(self.__magic_determine_other(other) ** self.data)

    def __deepcopy__(self, memo):
        """deepcopy backend data"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, (SparkSession, SparkDF, ps.DataFrame)):
                setattr(result, k, v)
                memo[id(v)] = v
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return self.data.__repr__()

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        return self.data._repr_html_()

    def _display_head_tail(
        self,
        rows_display_limit: int,
        cols_display_limit: int,
        n_cols: int,
        n_rows: int,
        tail: bool = False,
    ) -> pd.DataFrame:
        """Generate preview of head or tail rows with column truncation.
        
        Creates a pandas DataFrame for display purposes, handling large
        DataFrames by showing only specified row/column limits with ellipsis
        for truncated sections.
        
        Args:
            rows_display_limit (int): Maximum rows to display.
            cols_display_limit (int): Maximum columns to show per side when truncating.
            n_cols (int): Total number of columns in the DataFrame.
            n_rows (int): Total number of rows in the DataFrame.
            tail (bool): If True, return tail rows; otherwise return head rows.
            
        Returns:
            pd.DataFrame: Local pandas DataFrame for display, with truncated
                columns indicated by "..." column if necessary.
        """
        if tail:
            head_tail = self.data.tail(rows_display_limit).to_pandas()
            head_tail.index = [
                (n_rows - rows_display_limit + i) for i in range(rows_display_limit)
            ]
        else:
            head_tail = self.data.head(rows_display_limit).to_pandas()

        if n_cols > 2 * cols_display_limit:
            left_cols = self.columns[:cols_display_limit]
            right_cols = self.columns[-cols_display_limit:]
            tmp = pd.DataFrame(
                [["..."] for _ in range(len(head_tail))],
                index=head_tail.index,
                columns=["..."],
            )

            return pd.concat(
                [head_tail.loc[:, left_cols], tmp, head_tail.loc[:, right_cols]], axis=1
            )
        else:
            return head_tail

    def get_values(self, 
                   row: str | None = None, 
                   column: str | None = None) -> ScalarType | Sequence[ScalarType] | "SparkNavigation" | ps.Series:
        """Retrieve values by label-based indexing.
        
        Args:
            row (str | None): Row label for selection. If None, selects all rows.
            column (str | None): Column name for selection. If None, selects all columns.
            
        Returns:
            ScalarType | Sequence[ScalarType] | SparkNavigation | ps.Series: 
                - Scalar value if both row and column specified
                - Series of values if only column specified
                - DataFrame subset if only row specified
                - Full wrapped result if neither specified
                
        Note:
            Converts to local pandas for large results, subject to 
            PANDAS_CONVERSION_LIMIT.
        """
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
                    column: int | None = None) -> ScalarType | Sequence[ScalarType] | "SparkNavigation" | ps.Series:
        """Retrieve values by integer-position-based indexing.
        
        Args:
            row (int | None): Row integer position for selection. If None, selects all rows.
            column (int | None): Column integer position for selection. If None, selects all columns.
            
        Returns:
            ScalarType | Sequence[ScalarType] | SparkNavigation | ps.Series: 
                Selected values using iloc-style indexing, with same return
                type behavior as get_values().
                
        Note:
            Converts to local pandas for large results, subject to 
            PANDAS_CONVERSION_LIMIT.
        """
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
        """Create a new empty SparkNavigation with specified structure.
        
        Args:
            index (Iterable[Any] | None): Index labels for the empty DataFrame.
            columns (Iterable[str] | None): Column names for the empty DataFrame.
            
        Returns:
            SparkNavigation: New instance with empty data but defined structure.
        """
        return self._wrap_result(ps.DataFrame(index=index, columns=columns))

    @property
    def index(self) -> ps.Index:
        """Return the index of the underlying DataFrame."""
        return self.data.index
    
    def reset_index(self, 
                    drop: bool = False,
                    inplace: bool = False,
                    **kwargs) -> "SparkNavigation" | None:
        """Reset the index to default integer index.
        
        Args:
            drop (bool): If True, drop the current index instead of adding as column.
            inplace (bool): Ignored; always returns new instance for consistency.
            **kwargs: Additional arguments passed to underlying reset_index.
            
        Returns:
            SparkNavigation | None: New instance with reset index, or None if 
                inplace were supported (currently always returns new instance).
        """
        kwargs['inplace'] = False
        
        result = self.data.reset_index(drop=drop, **kwargs)
        return self._wrap_result(result)

    @property
    def columns(self) -> list[str]:
        """Return list of column names."""
        return self.data.columns.tolist()

    # @property
    # def session(self):
    #     return self.session

    @property
    def shape(self) -> tuple[int, int]:
        """Return tuple of (rows, columns) dimensions."""
        return self.data.shape

    @property
    def labels_dict(self):
        raise NotImplementedError("Method labels_dict not implemented for SparkNavigation.")

    def _get_column_index(self, column_name: Sequence[str] | str) -> int | list[int]:
        """Get integer position(s) of column name(s).
        
        Args:
            column_name (Sequence[str] | str): Column name or list of names.
            
        Returns:
            int | list[int]: Integer position(s) of the specified column(s).
            
        Raises:
            ValueError: If column_name type is not str or list.
        """
        if isinstance(column_name, str):
            return self.data.columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return self.data.columns.get_indexer(column_name)
        else:
            raise ValueError("Wrong column_name type.")

    def get_column_type(self, column_name: str | Iterable[str] | None = None) -> dict[str, type] | type | None:
        """Get Python type(s) corresponding to Spark schema type(s).
        
        Maps Spark SQL data types to native Python types using SparkTypeMapper.
        
        Args:
            column_name (str | Iterable[str] | None): Single column name, list of 
                names, or None for all columns.
                
        Returns:
            dict[str, type] | type | None: 
                - Single type if column_name is str
                - Dict mapping column names to types if column_name is iterable or None
                - None if column not found
        """
        spark_schema = self.data.to_spark().schema

        if isinstance(column_name, str):
            field = next(
                (f for f in spark_schema.fields if f.name == column_name), None
            )
            return SparkTypeMapper.to_python(field.dataType) if field else None

        result = {}
        target_cols = column_name if column_name is not None else self.data.columns
        for col in target_cols:
            field = next((f for f in spark_schema.fields if f.name == col), None)
            result[col] = SparkTypeMapper.to_python(field.dataType) if field else object
        
        return result           

    def astype(self, 
               dtype: dict[str, type], 
               errors: Literal["raise", "ignore"] = "raise") -> "SparkNavigation":
        """Cast columns to specified data types.
        
        Args:
            dtype (dict[str, type]): Mapping of column names to target Python types.
            errors (Literal["raise", "ignore"]): Error handling strategy.
            
        Returns:
            SparkNavigation: New instance with casted column types.
        """
        return self._wrap_result(self.data.astype(dtype=dtype))

    def update_column_type(self,
                           dtype: dict[str, type],
                           errors: Literal["raise", "ignore"] = "raise") -> SparkNavigation:
        """Update column types with validation and error handling.
        
        More robust than astype(), with checks for missing columns and null-only columns.
        
        Args:
            dtype (dict[str, type]): Mapping of column names to target Python types.
            errors (Literal["raise", "ignore"]): Whether to raise on conversion errors.
            
        Returns:
            SparkNavigation: Instance with updated column types.
            
        Raises:
            KeyError: If column not found and errors="raise".
            ValueError: If column contains only null values and errors="raise".
        """
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
        """Add a new column to the dataset.
        
        Args:
            data (Sequence[Any]): Column values to add.
            name (str | list[str]): Column name(s). Single-element lists are unwrapped.
            index (Sequence[Any] | None): Optional index for the new column.
            
        Note:
            Modifies self.data in place. Handles pyspark.pandas DataFrame/Series
            inputs with automatic joining.
        """
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
               axis: int = 0) -> "SparkNavigation":
        """Concatenate other datasets along specified axis.
        
        Args:
            other (Sequence[SparkNavigation]): List of SparkNavigation instances to append.
            reset_index (bool): If True, reset index in result to default integer index.
            axis (int): Axis along which to concatenate (0=rows, 1=columns).
            
        Returns:
            SparkNavigation: New instance with combined data.
        """
        new_data = ps.concat([self.data] + [d.data for d in other], axis=axis)
        if reset_index:
            new_data = new_data.reset_index(drop=True)
        return self._wrap_result(new_data)

    def from_dict(self, 
                  data: FromDictTypes, 
                  index: Iterable[Any] | Sized | None = None):
        """Load data from dictionary format.
        
        Args:
            data (FromDictTypes): Dictionary or record-style data to load.
            index (Iterable[Any] | Sized | None): Optional index to assign.
            
        Returns:
            SparkNavigation: Self, for method chaining.
        """
        if isinstance(data, dict):
            self.data = ps.DataFrame().from_records(data, columns=list(data.keys()))
        else:
            self.data = ps.DataFrame().from_records(data)
        if index is not None:
            self.data.index = index
        return self

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert dataset to dictionary format with data and index.
        
        Returns:
            dict[str, list[Any]]: Dictionary with keys:
                - "data": dict mapping column names to value lists
                - "index": list of index values
                
        Note:
            Subject to PANDAS_CONVERSION_LIMIT for large datasets.
        """
        self._check_pandas_conversion(obj=self.data, context="to_dict")
        pdf = self.data.to_pandas()
        return {
            "data": {column: pdf[column].to_list() for column in pdf.columns},
            "index": list(pdf.index),
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Convert dataset to list of row dictionaries.
        
        Returns:
            list[dict[str, Any]]: List where each element is a dict representing
                one row with column names as keys.
                
        Note:
            Subject to PANDAS_CONVERSION_LIMIT for large datasets.
        """
        self._check_pandas_conversion(obj=self.data, context="to_records")
        return self.data.to_pandas().to_dict(orient="records")

    def loc(self, items: Iterable[Any]) -> "SparkNavigation":
        """Label-based selection of rows.
        
        Args:
            items (Iterable[Any]): Labels of rows to select.
            
        Returns:
            SparkNavigation: Selected rows (note: currently returns raw data, 
                consider wrapping for consistency).
        """
        data = self.data.loc[items]
        return data

    def iloc(self, items: Iterable[Any]) -> "SparkNavigation":
        """Integer-position-based selection of rows.
        
        Args:
            items (Iterable[Any]): Integer positions or slice of rows to select.
            
        Returns:
            SparkNavigation: Selected rows, wrapped as DataFrame if needed.
        """
        if isinstance(items, int): 
            data = self.data.iloc[[items]]
        else:
            data = self.data.iloc[items]
        if not isinstance(data, ps.DataFrame):
            data = ps.DataFrame(data)
        return data


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    """Calculation-focused interface for PySpark-backed datasets.
    
    Extends SparkNavigation with statistical, aggregation, and analytical methods
    for distributed data processing. Provides pandas-like API for common data
    science operations while leveraging Spark's distributed computing capabilities.
    
    Inherits all navigation and indexing capabilities from SparkNavigation.
    """
    
    @staticmethod
    def _convert_agg_result(result: ps.Series | ps.DataFrame) -> "SparkDataset" | float:
        """Convert aggregation results to appropriate return type.
        
        Handles edge case where single-value aggregations should return scalar
        rather than DataFrame for convenience.
        
        Args:
            result (ps.Series | ps.DataFrame): Result from aggregation operation.
            
        Returns:
            SparkDataset | float: Wrapped DataFrame for multi-value results,
                or scalar float for single-value results.
        """
        if isinstance(result, ps.Series):
            result = result.to_frame()
        if result.shape == (1, 1):
            return float(result.to_spark().collect()[0][0])
        return result if isinstance(result, ps.DataFrame) else ps.DataFrame(result)

    def __init__(self, 
                 data: ps.DataFrame | dict | str | ps.Series | None = None,
                 session: SparkSession | None = None):
        """Initialize SparkDataset instance.
        
        Args:
            data (ps.DataFrame | dict | str | ps.Series | None): Source data.
            session (SparkSession | None): Spark session for operations.
        """
        super().__init__(data=data, session=session)

    def get(self, key: str, default: Any=None) -> Any:
        """Get column by name with optional default value.
        
        Args:
            key (str): Column name to retrieve.
            default (Any): Value to return if column not found.
            
        Returns:
            Any: Column data or default value.
        """
        return self.data.get(key, default)

    def take(self, 
             indices: int | Sequence[int], 
             axis: Literal["index", "columns", "rows"] | int = 0) -> "SparkDataset" | ps.Series:
        """Select elements at specified integer positions.
        
        Args:
            indices (int | Sequence[int]): Position(s) to select.
            axis (Literal["index", "columns", "rows"] | int): Axis for selection 
                (0/index/rows for row selection, 1/columns for column selection).
                
        Returns:
            SparkDataset | ps.Series: Selected data, wrapped appropriately.
        """
        if isinstance(indices, slice) and (axis == 1):
            self._wrap_result(self.data.iloc[indices])
        return self._wrap_result(self.data.take(indices=indices, axis=axis))

    def apply(self, func: Callable[..., Any], **kwargs) -> SparkDataset:
        """Apply function along axis of DataFrame.
        
        Args:
            func (Callable[..., Any]): Function to apply to each column/row.
            column_name (str, optional): Name for result column if func returns Series.
            **kwargs: Additional arguments passed to underlying apply.
            
        Returns:
            SparkDataset: Result with applied function, ensuring DataFrame output.
        """
        single_column_name = kwargs.pop("column_name", None)
        result = self.data.apply(func, **kwargs)
        if not isinstance(result, ps.DataFrame):
            result = result.to_frame(name=single_column_name)
        return self._wrap_result(result)

    def map(self, func: Callable[..., Any], na_action: Any = None, **kwargs) -> SparkDataset:
        """Map function over Series elements.
        
        Args:
            func (Callable[..., Any]): Function to apply element-wise.
            na_action (Any): Handling for NA values.
            **kwargs: Additional arguments for map operation.
            
        Returns:
            SparkDataset: Result with mapped values.
        """
        """Map function over Series elements."""
        if len(self.data.columns) == 1:
            col_name = self.data.columns[0]
            result = self.data[col_name].map(func, na_action=na_action, **kwargs)
            return self._wrap_result(result.to_frame())
        else:
            return self._wrap_result(self.data.apply(func, **kwargs))

    def is_empty(self) -> bool:
        """Check if dataset contains no data.
        
        Returns:
            bool: True if DataFrame has zero rows or columns.
        """
        return self.data.empty

    def unique(self) -> dict[str, list[Any]]:
        """Get unique values for each column.
        
        Returns:
            dict[str, list[Any]]: Mapping of column names to lists of unique values.
        """
        return {column: self.data[column].unique() for column in self.data.columns}

    def nunique(self, dropna: bool = True)-> dict[str, int]:
        """Count unique values for each column.
        
        Args:
            dropna (bool): Whether to exclude null values from count.
            
        Returns:
            dict[str, int]: Mapping of column names to unique value counts.
        """
        return {column: self.data[column].nunique() for column in self.data.columns}

    def groupby(self, by: str | Iterable[str], **kwargs) -> ps.groupby.GroupBy:
        """Group DataFrame by specified column(s).
        
        Args:
            by (str | Iterable[str]): Column name(s) to group by.
            **kwargs: Additional arguments for groupby operation.
            
        Returns:
            ps.groupby.GroupBy: GroupBy object for subsequent aggregation.
        """
        return self.data.groupby(by=by, **kwargs)

    def iter_groups(self, by: list[str]):
        """Iterate over groups defined by column(s).
        
        Args:
            by (list[str]): Column names defining group keys.
            
        Yields:
            tuple: (group_key, SparkNavigation) for each unique combination 
                of grouping column values.
        """
        keys_df = self.data[by].drop_duplicates().to_pandas()
        for _, row in keys_df.iterrows():
            key = row[by[0]] if len(by) == 1 else tuple(row[col] for col in by)
            mask = None
            for col in by:
                col_mask = self.data[col] == row[col]
                mask = col_mask if mask is None else mask & col_mask
            yield key, self.data[mask]
    
    def count_groups(self, group_cols: list[str]) -> int:
        """Count unique combinations of group_cols"""
        if not group_cols:
            return 1
        return int(len(self.data[group_cols].drop_duplicates()))


    def grouped_value_counts(self, by: list[str], feature_cols: list[str] | None=None):
        from functools import reduce
        
        if feature_cols is None:
            feature_cols = [col for col in self.data.columns if col not in set(by)]
        sdf = self.data.to_spark()
        agg_sdfs = []
        for col in feature_cols:
            agg_sdf = (
                sdf.filter(F.col(col).isNotNull())
                .groupBy(*by, col)
                .count()
                .groupBy(*by)
                .agg(
                    F.map_from_entries(
                        F.collect_list(F.struct(F.col(col), F.col("count")))
                    ).alias(col)
                )
            )
            agg_sdfs.append(agg_sdf)

        result_sdf = reduce(lambda a, b: a.join(b, on=by, how="outer"), agg_sdfs)
        result_psdf = result_sdf.pandas_api().set_index(by[0] if len(by) == 1 else by)
        rows = result_psdf.to_dict(orient="index")
        return {
            "data": {col: [rows[k][col] for k in rows] for col in feature_cols},
            "index": list(rows.keys()),
        }

    def agg(self, func: str | list, **kwargs) -> SparkDataset | float:
        """Aggregate data using specified function(s).
        
        Automatically selects numeric columns if subset not specified.
        
        Args:
            func (str | list): Aggregation function name(s) (e.g., "sum", ["mean", "std"]).
            subset (str | list, optional): Column(s) to aggregate. Defaults to numeric columns.
            **kwargs: Additional arguments for aggregation.
            
        Returns:
            SparkDataset | float: Aggregated results, or scalar for single-value results.
        """
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
        """Compute maximum value(s)."""
        return self._wrap_result(self.agg(["max"]))

    def idxmax(self) -> SparkDataset | float:
        """Return index of maximum value(s)."""
        return self._wrap_result(self.agg(["idxmax"]))

    def min(self) -> SparkDataset | float:
        """Compute minimum value(s)."""
        return self._wrap_result(self.agg(["min"]))

    def count(self) -> SparkDataset | float:
        """Count non-null values."""
        return self._wrap_result(self.agg(["count"]))

    def sum(self) -> SparkDataset | float:
        """Compute sum of values."""
        return self._wrap_result(self.agg(["sum"]))

    def mean(self) -> SparkDataset | float:
        """Compute arithmetic mean."""
        return self._wrap_result(self.agg("mean"))

    def mode(self, numeric_only: bool = False, dropna: bool = True) -> SparkDataset:
        """Compute mode (most frequent value) for each column.
        
        Args:
            numeric_only (bool): Consider only numeric columns.
            dropna (bool): Exclude null values from calculation.
            
        Returns:
            SparkDataset: DataFrame containing mode values.
        """
        return self._wrap_result(self.data.mode(numeric_only=numeric_only, dropna=dropna))

    def std(self, skipna: bool = True, ddof: int = 1) -> "SparkDataset" | float:
        """Compute sample standard deviation.
        
        Args:
            skipna (bool): Exclude null values.
            ddof (int): Delta degrees of freedom (default 1 for sample std).
            
        Returns:
            SparkDataset | float: Standard deviation values.
        """
        result = self.data.std()
        converted = self._convert_agg_result(
            result.to_frame() if isinstance(result, ps.Series) else result
        )
        return self._wrap_result(converted)

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False) -> SparkDataset | float:
        """Compute variance.
        
        Args:
            skipna (bool): Exclude null values.
            ddof (int): Delta degrees of freedom.
            numeric_only (bool): Consider only numeric columns.
            
        Returns:
            SparkDataset | float: Variance values.
        """
        result = self.data.var()
        converted = self._convert_agg_result(
            result.to_frame() if isinstance(result, ps.Series) else result
        )
        return self._wrap_result(converted)

    def log(self) -> SparkDataset:
        """Compute natural logarithm of numeric values.
        
        Returns:
            SparkDataset: DataFrame with log-transformed values.
        """
        np_data = np.log(self.data.to_numpy())
        return self._wrap_result(ps.DataFrame(np_data, columns=self.data.columns))

    def cov(self) -> SparkDataset:
        """Compute covariance matrix for numeric columns.
        
        Returns:
            SparkDataset: Covariance matrix, or None if no numeric columns.
        """
        numeric_cols = self.get_numeric_columns()        
        if len(numeric_cols) == 0:
            return None

        result = self.data[numeric_cols].cov()
        return self._wrap_result(result)

    def quantile(self, q: float = 0.5) -> "SparkDataset" | float:
        """Compute quantile(s) for numeric columns.
        
        Args:
            q (float): Quantile value(s) to compute (0 <= q <= 1).
            
        Returns:
            SparkDataset | float: Quantile values.
        """
        if isinstance(q, list) and len(q) > 1:
            return self.data.quantile(q=q)
        else:
            result = self.data.quantile(q=q)
            converted = self._convert_agg_result(
                result.to_frame() if isinstance(result, ps.Series) else result
            )
            if isinstance(converted, ps.DataFrame):
                return converted
            return self._wrap_result(converted)

    def coefficient_of_variation(self) -> "SparkDataset" | float:
        """Compute coefficient of variation (std/mean) for numeric columns.
        
        Returns:
            SparkDataset | float: CV values, or None if no numeric columns.
                Handles division by zero by replacing with NaN.
        """
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

    def sort_index(self, ascending: bool = True, **kwargs) -> "SparkDataset":
        """Sort dataset by index labels.
        
        Args:
            ascending (bool): Sort in ascending order.
            **kwargs: Additional arguments for sort_index.
            
        Returns:
            SparkDataset: Sorted dataset.
        """
        return self._wrap_result(self.data.sort_index(ascending=ascending, **kwargs))

    def get_numeric_columns(self) -> list[str]:
        """Identify columns with numeric data types.
        
        Returns:
            list[str]: List of column names with numeric types.
        """
        types = self.get_column_type()
        return [
            col for col, dtype in types.items()
            if dtype in [int, float, np.int64, np.float64, np.int32, np.float32]
        ]

    def corr(self, numeric_only: bool = False) -> "SparkDataset" | float:
        """Compute Pearson correlation matrix for numeric columns.
        
        Args:
            numeric_only (bool): Currently ignored; only numeric columns processed.
            
        Returns:
            SparkDataset | float: Correlation matrix, or None if no numeric columns.
        """
        numeric_cols = self.get_numeric_columns()

        if len(numeric_cols) == 0:
            return None

        result = self.data[numeric_cols].corr(method="pearson")

        if isinstance(result, ps.DataFrame):
            return self._wrap_result(result)
        return result

    def isna(self) -> "SparkDataset":
        """Detect missing values.
        
        Returns:
            SparkDataset: Boolean DataFrame indicating null positions.
        """
        return self._wrap_result(self.data.isna())

    def sort_values(self, by: str | list[str], ascending: bool = True, **kwargs) -> "SparkDataset":
        """Sort by values in specified column(s).
        
        Args:
            by (str | list[str]): Column name(s) to sort by.
            ascending (bool): Sort in ascending order.
            **kwargs: Additional arguments for sort_values.
            
        Returns:
            SparkDataset: Sorted dataset.
        """
        return self._wrap_result(self.data.sort_values(by=by, ascending=ascending, **kwargs))

    def value_counts(self,
                     normalize: bool = False,
                     sort: bool = True,
                     ascending: bool = False,
                     dropna: bool = True) -> "SparkDataset":
        """Count unique values in first column.
        
        Args:
            normalize (bool): Return proportions instead of counts.
            sort (bool): Sort results by count.
            ascending (bool): Sort in ascending order if sorting.
            dropna (bool): Exclude null values from counts.
            
        Returns:
            SparkDataset: DataFrame with value counts, reset to column format.
        """
        
        col = list(self.data.columns)[0]
        series = self.data[col]

        result = series.value_counts(
            normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
        )

        result_df = result.to_frame(name="count").reset_index()

        result_df = result_df.rename(columns={"index": col})

        return self._wrap_result(result_df)

    def na_counts(self) -> "SparkDataset" | int:
        """Count null values per column.
        
        Returns:
            SparkDataset | int: Null counts as DataFrame, or scalar for single-column case.
        """
        data = self.data.isna().sum().to_frame().T

        if data.shape[0] == 1 and data.shape[1] == 1:
            return int(data.to_spark().collect()[0][0])

        old_index_name = data.index.tolist()[0]

        return self._wrap_result(data.rename(index={old_index_name: "na_counts"}))

    def dot(self, other: 'SparkDataset' | np.ndarray | pd.DataFrame) -> "SparkDataset" | float:
        """Compute dot product with another dataset or array.
        
        Handles multiple input types with appropriate dimension validation.
        
        Args:
            other (SparkDataset | np.ndarray | pd.DataFrame): Right-hand operand.
            
        Returns:
            SparkDataset | float: Dot product result.
            
        Raises:
            ValueError: If dimensions are incompatible.
            TypeError: If other is unsupported type.
        """
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
            return self._wrap_result(
                result if isinstance(result, ps.DataFrame) else result.to_frame()
            )

        elif isinstance(other, pd.DataFrame):
            other_ps = ps.DataFrame(other)
            if other_ps.shape[0] != len(self.data.columns):
                raise ValueError(
                    f"Matrix dimensions not aligned: {self.data.shape} dot {other_ps.shape}"
                )
            other_ps.index = self.data.columns
            result = self.data.dot(other_ps)
            return self._wrap_result(
                result if isinstance(result, ps.DataFrame) else result.to_frame()
            )

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
                return self._wrap_result(
                    float(result)
                    if isinstance(result, (int, float, np.number))
                    else result
                )
            else:
                result = (self_subset * other_subset).sum()
                return self._wrap_result(
                    result if isinstance(result, ps.DataFrame) else result.to_frame()
                )

        else:
            raise TypeError(
                f"Unsupported type for dot: {type(other)}. "
                f"Expected SparkDataset, np.ndarray, or pd.DataFrame"
            )


    def dropna(self,
               how: Literal["any", "all"] = "any",
               subset: str | Iterable[str] | None = None,
               axis: Literal["index", "rows", "columns"] | int = 0) -> SparkDataset:
        """Remove missing values.
        
        Args:
            how (Literal["any", "all"]): Require any or all values to be NA to drop.
            subset (str | Iterable[str] | None): Columns to consider for NA check.
            axis (Literal["index", "rows", "columns"] | int): Axis to drop from.
            
        Returns:
            SparkDataset: Dataset with specified NA values removed.
        """
        return self._wrap_result(self.data.dropna(how=how, subset=subset, axis=axis))

    def transpose(self, names: Sequence[str] | None = None) -> "SparkDataset":
        """Transpose rows and columns.
        
        Args:
            names (Sequence[str] | None): Optional new column names after transpose.
            
        Returns:
            SparkDataset: Transposed dataset.
        """
        result = self.data.transpose()
        if names is not None:
            result.columns = names
        return self._wrap_result(
            result if isinstance(result, ps.DataFrame) else ps.DataFrame(result)
        )

    @staticmethod
    def _reproducible_sample(df: ps.DataFrame,
                             n: int = None,
                             frac: float = None,
                             replace: bool = False,
                             seed: int = 42) -> ps.DataFrame:
        """Generate reproducible random sample using Spark hashing.
        
        Uses deterministic hash-based shuffling for reproducible sampling
        across Spark executions.
        
        Args:
            df (ps.DataFrame): Source DataFrame.
            n (int | None): Number of rows to sample.
            frac (float | None): Fraction of rows to sample.
            replace (bool): Sample with replacement.
            seed (int): Random seed for reproducibility.
            
        Returns:
            ps.DataFrame: Sampled DataFrame with original index preserved.
        """
        
        df_with_index = df.reset_index()
        index_cols = df_with_index.columns[: df.index.nlevels]

        df_with_index["_shuffle_key"] = F.abs(
            F.hash(*[F.col(col) for col in index_cols], F.lit(seed))
        )

        shuffled = df_with_index.sort_values("_shuffle_key")

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

        sampled = sampled.drop(columns=["_shuffle_key"])
        if df.index.nlevels == 1:
            sampled = sampled.set_index(index_cols[0])
        else:
            sampled = sampled.set_index(index_cols)

        return sampled

    def sample(self,
               frac: float | None = None,
               n: int | None = None,
               random_state: int | None = None,
               method: Literal["approx", "exact"] = "exact") -> "SparkDataset":
        """Generate random sample of dataset.
        
        Currently uses reproducible hash-based sampling for consistency.
        
        Args:
            frac (float | None): Fraction of rows to sample.
            n (int | None): Exact number of rows to sample.
            random_state (int | None): Seed for reproducibility.
            method (Literal["approx", "exact"]): Currently ignored; uses exact method.
            
        Returns:
            SparkDataset: Sampled dataset.
        """
        
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
                      exclude: str | None = None) -> "SparkDataset":
        """Select columns based on data type.
        
        Args:
            include (str | None): Type(s) to include.
            exclude (str | None): Type(s) to exclude.
            
        Returns:
            SparkDataset: Dataset with filtered columns.
        """
        return self._wrap_result(self.data.select_dtypes(include=include, exclude=exclude))

    def isin(self, values: Iterable) -> SparkDataset:
        """Test if elements are contained in provided values.
        
        Args:
            values (Iterable): Collection of values to test against.
            
        Returns:
            SparkDataset: Boolean DataFrame indicating membership.
        """
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
        """Merge with another dataset using database-style join.
        
        Args:
            right (SparkDataset): Dataset to merge with.
            on (str | None): Column name(s) to join on (must exist in both).
            left_on (str | None): Column(s) from left dataset to join on.
            right_on (str | None): Column(s) from right dataset to join on.
            left_index (bool | None): Use left index as join key.
            right_index (bool | None): Use right index as join key.
            suffixes (tuple[str, str]): Suffixes for overlapping column names.
            how (Literal["left", "right", "inner", "outer", "cross"]): Join type.
            
        Returns:
            SparkDataset: Merged dataset.
            
        Raises:
            MergeOnError: If join keys not found in datasets.
        """
        for on_ in [on, left_on, right_on]:
            if on_ and (
                on_ not in [*self.columns, *right.columns]
                if isinstance(on_, str)
                else any(c not in [*self.columns, *right.columns] for c in on_)
            ):
                raise MergeOnError(on_)
        if not all(
            [
                on,
                left_on,
                right_on,
            ]
        ) and all([left_index is None, right_index is None]):
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
             columns: str | Iterable[str] | None = None) -> "SparkDataset":
        """Drop specified labels from rows or columns.
        
        Args:
            labels (str | None): Labels to drop.
            axis (int | None): Axis to drop from (0=index, 1=columns).
            columns (str | Iterable[str] | None): Alternative way to specify columns to drop.
            
        Returns:
            SparkDataset: Dataset with specified labels removed.
        """
        return self._wrap_result(self.data.drop(labels=labels, axis=axis, columns=columns))

    def filter(self,
               items: list | None = None,
               regex: str | None = None,
               axis: int | str = 0) -> "SparkDataset":
        """Filter columns based on labels or regex pattern.
        
        Args:
            items (list | None): List of column labels to keep.
            regex (str | None): Regex pattern to match column names.
            axis (int | str): Axis to filter (currently only column filtering supported).
            
        Returns:
            SparkDataset: Filtered dataset.
        """
        return self._wrap_result(self.data.filter(items=items, regex=regex, axis=axis))

    def rename(self, columns: dict[str, str]) -> "SparkDataset":
        """Rename columns using mapping dictionary.
        
        Args:
            columns (dict[str, str]): Mapping of old names to new names.
            
        Returns:
            SparkDataset: Dataset with renamed columns.
        """
        return self._wrap_result(self.data.rename(columns=columns))

    def replace(self, to_replace: Any = None, value: Any = None, regex: bool = False) -> "SparkDataset":
        """Replace values matching criteria.
        
        Args:
            to_replace (Any): Value(s) to replace.
            value (Any): Replacement value(s).
            regex (bool): Treat to_replace as regex pattern.
            
        Returns:
            SparkDataset: Dataset with replaced values.
        """
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
        """Conform dataset to new index with optional fill value.
        
        Args:
            labels (str): New index labels (note: parameter type may need review).
            fill_value (str | None): Value to use for newly missing entries.
            
        Returns:
            SparkDataset: Reindexed dataset.
        """
        return self._wrap_result(self.data.reindex(labels, fill_value=fill_value))
    
    def fillna(self,
               values: ScalarType | dict[str, ScalarType] | None = None,
               method: Literal["bfill", "ffill"] | None = None,
               **kwargs) -> SparkDataset:
        """Fill missing values using specified strategy.
        
        Args:
            values (ScalarType | dict[str, ScalarType] | None): Value(s) to fill NA.
            method (Literal["bfill", "ffill"] | None): Fill method (backward/forward).
            **kwargs: Additional arguments for fillna.
            
        Returns:
            SparkDataset: Dataset with filled missing values.
            
        Raises:
            ValueError: If method is not "bfill" or "ffill".
        """
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

    def list_to_columns(self, column: str) -> "SparkDataset":
        """Expand list-valued column into multiple separate columns.
        
        Splits a column containing lists into separate columns for each
        list element position.
        
        Args:
            column (str): Name of column containing list values.
            
        Returns:
            SparkDataset: Dataset with expanded columns named {column}_0, {column}_1, etc.
        """
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
