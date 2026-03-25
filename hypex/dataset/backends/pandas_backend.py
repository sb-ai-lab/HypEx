from __future__ import annotations

from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Sequence,
    Sized,
    Union,
    Optional,
    List,
)
from typing_extensions import Self

import numpy as np
import pandas as pd  # type: ignore
import pyspark.sql as spark
import pyspark.sql.functions as F

import pyspark.pandas as ps

from pyspark.ml.feature import StringIndexer

from ...utils import FromDictTypes, MergeOnError, ScalarType
from ...utils.adapter import Adapter
from ...utils.constants import UTILITY_INDEX_COL_NAME
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class PandasNavigation(DatasetBackendNavigation):

    def _wrap_result(
        self, result: pd.DataFrame | pd.Series | Any, wrap_series: bool = False
    ) -> "PandasNavigation" | pd.Series | Any:
        if isinstance(result, pd.DataFrame):
            return self.__class__(data=result)

        if wrap_series and isinstance(result, pd.Series):
            return self.__class__(data=result.to_frame())

        return result

    def _data_compression(
        self,
        data: spark.DataFrame,
        data_compression: Literal["downcasting", "encoding", "auto", "disable"],
        non_compresion_cols: List[str] | None,
    ) -> pd.DataFrame:
        """Compress data before convertation `spark.DataFrame` to pandas.DataFrame.

        Args:
            data: `spark.DataFrame data` copressing data.
            data_compression: `Literal["downcasting", "encoding", "auto", "disable"]` compression mode.
            non_compresion_cols: `List[str] | None` list of columns that shouldn't be encoded.

        Returns:
            `pd.DataFrame`: compressed dataframe.
        """
        columns_dict = dict(data.dtypes)
        labels = {}
        result = data

        if data_compression in ["downcasting", "auto"]:
            double_columns = [
                col
                for col, c_type in columns_dict.items()
                if c_type == "double" or c_type.startswith("decimal")
            ]
            result = self._downcasting(data, double_columns)

        if data_compression in ["encoding", "auto"]:
            if non_compresion_cols is None:
                non_compresion_cols = []
            categorical_columns = [
                col
                for col, c_type in columns_dict.items()
                if c_type in ["string", "varchar"] and col not in non_compresion_cols
            ]
            result, labels = self._encoding(result, categorical_columns)

        self._labels_dict = labels
        return result.toPandas()

    @staticmethod
    def _encoding(
        data: spark.DataFrame, categorical_columns: List[str]
    ) -> spark.DataFrame:
        """Encoding categorical features.

        Args:
            data: `spark.DataFrame data` copressing data.
            categorical_columns: `List[str]` list of columns for encoding.

        Returns:
            `spark.DataFrame`: dataframe with encoded categorical columns.
        """
        filled_data = data.na.fill("UNKNOWN")
        indexer = StringIndexer(
            inputCols=categorical_columns,
            outputCols=[f"{col}_indexed" for col in categorical_columns],
            handleInvalid="keep",
        )
        model = indexer.fit(filled_data)
        labels = {
            col: {idx: label for idx, label in enumerate(labels_list)}
            for labels_list, col in zip(model.labelsArray, categorical_columns)
        }

        return (
            (
                model.transform(filled_data).select(
                    *[
                        (
                            F.col(f"{col}_indexed").cast("int").alias(col)
                            if col in categorical_columns
                            else F.col(col)
                        )
                        for col in filled_data.columns
                    ]
                )
            ),
            labels,
        )

    @staticmethod
    def _downcasting(
        data: spark.DataFrame, numeric_columns: List[str]
    ) -> spark.DataFrame:
        """Downcasting data.

        Args:
            data: `spark.DataFrame data` copressing data.
            numeric_columns: `List[str]` list of floating point columns.

        Returns:
            `spark.DataFrame`: downcasted dataframe.
        """
        return data.select(
            *[
                F.col(col).cast("float") if col in numeric_columns else F.col(col)
                for col in data.columns
            ]
        )

    @staticmethod
    def _read_file(filename: str | Path) -> pd.DataFrame:
        """Read a file into a pandas DataFrame based on file extension.

        Args:
            filename: Path to the file to read (string or Path object).

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            ValueError: If the file extension is not supported (.csv or .xlsx).
        """
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return pd.read_csv(filename)
        elif file_extension == ".xlsx":
            return pd.read_excel(filename)
        else:
            raise ValueError(f"Unsupported file extension {file_extension}")

    def __init__(
        self,
        data: pd.DataFrame | dict | str | pd.Series | None = None,
        data_compression: Literal[
            "downcasting", "encoding", "auto", "disable"
        ] = "auto",
        non_compresion_cols: List[str] | None = None,
    ):
        """Initialize PandasNavigation with various data sources.

        Args:
             Input data in one of the following formats:
                - pd.DataFrame: Used directly.
                - pd.Series: Converted to DataFrame.
                - spark.DataFrame: Converted via toPandas().
                - dict: Expected format {"data": ..., "index": ...} or {"data": ...}.
                - str: Path to a file (.csv or .xlsx) to be read.
                - None: Creates an empty DataFrame.

             data_compression: regime for compression `spark.DataFrame` to `pandas.DataFrame`.
        """
        self._labels_dict = {}
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
        elif isinstance(data, spark.DataFrame):
            self.data = self._data_compression(
                data, data_compression, non_compresion_cols
            )
        elif isinstance(data, ps.DataFrame):
            self.data = data.to_pandas()
        elif isinstance(data, dict):
            wrapped = {k: v if isinstance(v, list) else [v] for k, v in data["data"].items()}
            if "index" in data.keys():
                self.data = pd.DataFrame(data=wrapped, index=data["index"])
            else:
                self.data = pd.DataFrame(data=wrapped)
        elif isinstance(data, str):
            self.data = self._read_file(data)
        else:
            self.data = pd.DataFrame()

    def __getitem__(self, item):
        """Enable indexing/slicing access to underlying DataFrame.

        Supports:
            - Integer/slice: Row-based access via iloc.
            - str/list: Column selection.
            - pd.DataFrame: Boolean masking or column selection.

        Args:
            item: Indexer (int, slice, str, list, or pd.DataFrame).

        Returns:
            Selected data from the DataFrame.

        Raises:
            KeyError: If the specified column or row does not exist.
        """
        if isinstance(item, (slice, int)):
            return self.data.iloc[item]
        if isinstance(item, (str, list)):
            return self.data[item]
        if isinstance(item, pd.DataFrame):
            if len(item.columns) == 1:
                return self.data[item.iloc[:, 0]]
            else:
                return self.data[item]
        raise KeyError("No such column or row")

    def __len__(self):
        """Return the number of rows in the DataFrame.

        Returns:
            int: Number of rows.
        """
        return len(self.data)

    @staticmethod
    def __magic_determine_other(other) -> Any:
        """Extract raw pandas DataFrame from PandasDataset or return value as-is.

        Private helper for operator overloading to enable operations between
        PandasDataset instances and other types.

        Args:
            other: Object to convert (PandasDataset or any other type).

        Returns:
            pd.DataFrame if other is PandasDataset, otherwise other unchanged.
        """
        if isinstance(other, PandasDataset):
            return other.data
        else:
            return other

    # comparison operators:
    def __eq__(self, other) -> Any:
        """Element-wise equality comparison (==).

        Args:
            other: Value to compare against (PandasDataset or scalar/array-like).

        Returns:
            pd.DataFrame: Boolean DataFrame of comparison results.
        """
        return self._wrap_result(self.data == self.__magic_determine_other(other))

    def __ne__(self, other) -> Any:
        """Element-wise inequality comparison (!=).

        Args:
            other: Value to compare against.

        Returns:
            pd.DataFrame: Boolean DataFrame of comparison results.
        """
        return self._wrap_result(self.data != self.__magic_determine_other(other))

    def __le__(self, other) -> Any:
        """Element-wise less-than-or-equal comparison (<=).

        Args:
            other: Value to compare against.

        Returns:
            pd.DataFrame: Boolean DataFrame of comparison results.
        """
        return self._wrap_result(self.data <= self.__magic_determine_other(other))

    def __lt__(self, other) -> Any:
        """Element-wise less-than comparison (<).

        Args:
            other: Value to compare against.

        Returns:
            pd.DataFrame: Boolean DataFrame of comparison results.
        """
        return self._wrap_result(self.data < self.__magic_determine_other(other))

    def __ge__(self, other) -> Any:
        """Element-wise greater-than-or-equal comparison (>=).

        Args:
            other: Value to compare against.

        Returns:
            pd.DataFrame: Boolean DataFrame of comparison results.
        """
        return self._wrap_result(self.data >= self.__magic_determine_other(other))

    def __gt__(self, other) -> Any:
        """Element-wise greater-than comparison (>).

        Args:
            other: Value to compare against.

        Returns:
            pd.DataFrame: Boolean DataFrame of comparison results.
        """
        return self._wrap_result(self.data > self.__magic_determine_other(other))

    # Unary operations:
    def __pos__(self) -> Any:
        """Unary positive operation (+self).

        Returns:
            pd.DataFrame: Result of unary positive on underlying data.
        """
        return self._wrap_result(+self.data)

    def __neg__(self) -> Any:
        """Unary negation operation (-self).

        Returns:
            pd.DataFrame: Result of unary negation on underlying data.
        """
        return self._wrap_result(-self.data)

    def __abs__(self) -> Any:
        """Absolute value operation (abs(self)).

        Returns:
            pd.DataFrame: Element-wise absolute values.
        """
        return self._wrap_result(abs(self.data))

    def __invert__(self) -> Any:
        """Bitwise inversion operation (~self).

        Returns:
            pd.DataFrame: Element-wise bitwise NOT results.
        """
        return self._wrap_result(~self.data)

    def __round__(self, ndigits: int = 0) -> Any:
        """Round numeric values to specified decimal places.

        Args:
            ndigits: Number of decimal places (default: 0).

        Returns:
            pd.DataFrame: Rounded values.
        """
        return self._wrap_result(round(self.data, ndigits))

    # Binary operations:
    def __add__(self, other) -> Any:
        """Element-wise addition (self + other).

        Args:
            other: Value to add (PandasDataset, scalar, or array-like).

        Returns:
            pd.DataFrame: Result of addition.
        """
        return self._wrap_result(self.data + self.__magic_determine_other(other))

    def __sub__(self, other) -> Any:
        """Element-wise subtraction (self - other).

        Args:
            other: Value to subtract.

        Returns:
            pd.DataFrame: Result of subtraction.
        """
        return self._wrap_result(self.data - self.__magic_determine_other(other))

    def __mul__(self, other) -> Any:
        """Element-wise multiplication (self * other).

        Args:
            other: Value to multiply by.

        Returns:
            pd.DataFrame: Result of multiplication.
        """
        return self._wrap_result(self.data * self.__magic_determine_other(other))

    def __floordiv__(self, other) -> Any:
        """Element-wise floor division (self // other).

        Args:
            other: Divisor value.

        Returns:
            pd.DataFrame: Result of floor division.
        """
        return self._wrap_result(self.data // self.__magic_determine_other(other))

    def __div__(self, other) -> Any:
        """Element-wise division (self / other) - Python 2 compatibility.

        Args:
            other: Divisor value.

        Returns:
            pd.DataFrame: Result of division.
        """
        return self._wrap_result(self.data / self.__magic_determine_other(other))

    def __truediv__(self, other) -> Any:
        """Element-wise true division (self / other).

        Args:
            other: Divisor value.

        Returns:
            pd.DataFrame: Result of true division.
        """
        return self._wrap_result(self.data / self.__magic_determine_other(other))

    def __mod__(self, other) -> Any:
        """Element-wise modulo operation (self % other).

        Args:
            other: Divisor value.

        Returns:
            pd.DataFrame: Remainder after division.
        """
        return self._wrap_result(self.data % self.__magic_determine_other(other))

    def __pow__(self, other) -> Any:
        """Element-wise exponentiation (self ** other).

        Args:
            other: Exponent value.

        Returns:
            pd.DataFrame: Result of exponentiation.
        """
        return self._wrap_result(self.data ** self.__magic_determine_other(other))

    def __and__(self, other) -> Any:
        """Element-wise bitwise AND operation (self & other).

        Args:
            other: Value for bitwise AND.

        Returns:
            pd.DataFrame: Result of bitwise AND.
        """
        return self._wrap_result(self.data & self.__magic_determine_other(other))

    def __or__(self, other) -> Any:
        """Element-wise bitwise OR operation (self | other).

        Args:
            other: Value for bitwise OR.

        Returns:
            pd.DataFrame: Result of bitwise OR.
        """
        return self._wrap_result(self.data | self.__magic_determine_other(other))

    # Right arithmetic operators:
    def __radd__(self, other) -> Any:
        """Reflected addition (other + self).

        Args:
            other: Left-hand operand.

        Returns:
            pd.DataFrame: Result of addition.
        """
        return self._wrap_result(self.__magic_determine_other(other) + self.data)

    def __rsub__(self, other) -> Any:
        """Reflected subtraction (other - self).

        Args:
            other: Left-hand operand.

        Returns:
            pd.DataFrame: Result of subtraction.
        """
        return self._wrap_result(self.__magic_determine_other(other) - self.data)

    def __rmul__(self, other) -> Any:
        """Reflected multiplication (other * self).

        Args:
            other: Left-hand operand.

        Returns:
            pd.DataFrame: Result of multiplication.
        """
        return self._wrap_result(self.__magic_determine_other(other) * self.data)

    def __rfloordiv__(self, other) -> Any:
        """Reflected floor division (other // self).

        Args:
            other: Left-hand operand (dividend).

        Returns:
            pd.DataFrame: Result of floor division.
        """
        return self._wrap_result(self.__magic_determine_other(other) // self.data)

    def __rdiv__(self, other) -> Any:
        """Reflected division (other / self) - Python 2 compatibility.

        Args:
            other: Left-hand operand (dividend).

        Returns:
            pd.DataFrame: Result of division.
        """
        return self._wrap_result(self.__magic_determine_other(other) / self.data)

    def __rtruediv__(self, other) -> Any:
        """Reflected true division (other / self).

        Args:
            other: Left-hand operand (dividend).

        Returns:
            pd.DataFrame: Result of true division.
        """
        return self._wrap_result(self.__magic_determine_other(other) / self.data)

    def __rmod__(self, other) -> Any:
        """Reflected modulo (other % self).

        Args:
            other: Left-hand operand (dividend).

        Returns:
            pd.DataFrame: Remainder after division.
        """
        return self._wrap_result(self.__magic_determine_other(other) % self.data)

    def __rpow__(self, other) -> Any:
        """Reflected exponentiation (other ** self).

        Args:
            other: Base value.

        Returns:
            pd.DataFrame: Result of exponentiation.
        """
        return self._wrap_result(self.__magic_determine_other(other) ** self.data)

    def __repr__(self):
        """Return string representation of the underlying DataFrame.

        Returns:
            str: String representation.
        """
        return self._wrap_result(self.data.__repr__())

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebook display.

        Returns:
            str: HTML string representation.
        """
        return self._wrap_result(self.data._repr_html_())

    def _display_head_tail(
        self,
        rows_display_limit: int,
        cols_display_limit: int,
        n_cols: int,
        n_rows: int,
        tail: bool = False,
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
            If downcasting is applyed, repr substitutes encoded values to real one.
        """
        if tail:
            head_tail = self.data.tail(rows_display_limit)
            head_tail.index = [
                (n_rows - rows_display_limit + i) for i in range(rows_display_limit)
            ]
        else:
            head_tail = self.data.head(rows_display_limit)

        if n_cols > 2 * cols_display_limit:
            left_cols = self.columns[:cols_display_limit]
            right_cols = self.columns[-cols_display_limit:]
            tmp = pd.DataFrame(
                [["..."] for _ in range(len(head_tail))],
                index=head_tail.index,
                columns=["..."],
            )

            # return self._wrap_result(pd.concat([head_tail.loc[:, left_cols],
            #                          tmp,
            #                          head_tail.loc[:, right_cols]],
            #                          axis=1).replace(self.labels_dict))
            return pd.concat(
                [head_tail.loc[:, left_cols], tmp, head_tail.loc[:, right_cols]], axis=1
            ).replace(self.labels_dict)
        else:
            # return self._wrap_result(head_tail.replace(self.labels_dict))
            return head_tail.replace(self.labels_dict)

    def get_values(
        self,
        row: str | None = None,
        column: str | None = None,
    ) -> Any:
        """Get values by label-based indexing.

        Args:
            row: Row label (index value). If None, all rows selected.
            column: Column name. If None, all columns selected.

        Returns:
            Single value if both row and column specified,
            otherwise list of values from selected row/column/DataFrame.
        """
        if (column is not None) and (row is not None):
            return self._wrap_result(self.data.loc[row, column])
        elif column is not None:
            result = self.data.loc[:, column]
        elif row is not None:
            result = self.data.loc[row, :]
        else:
            result = self.data
        return self._wrap_result(result.values.tolist())

    def iget_values(
        self,
        row: int | None = None,
        column: int | None = None,
    ) -> Any:
        """Get values by integer-position-based indexing.

        Args:
            row: Integer row position. If None, all rows selected.
            column: Integer column position. If None, all columns selected.

        Returns:
            Single value if both row and column specified,
            otherwise list of values from selected row/column/DataFrame.
        """
        if (column is not None) and (row is not None):
            return self._wrap_result(self.data.iloc[row, column])
        elif column is not None:
            result = self.data.iloc[:, column]
        elif row is not None:
            result = self.data.iloc[row, :]
        else:
            result = self.data
        return self._wrap_result(result.values.tolist())

    def create_empty(
        self,
        index: Iterable | None = None,
        columns: Iterable[str] | None = None,
    ):
        """Replace current data with an empty DataFrame with specified structure.

        Args:
            index: Iterable of index labels.
            columns: Iterable of column names.

        Returns:
            self: Returns self for method chaining.
        """
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    @property
    def index(self):
        """Return the index (row labels) of the DataFrame.

        Returns:
            pd.Index: DataFrame index.
        """
        return self.data.index

    @property
    def columns(self):
        """Return the column labels of the DataFrame.

        Returns:
            pd.Index: DataFrame columns.
        """
        return self.data.columns

    @property
    def session(self):
        """Return session object (not applicable for pandas backend).

        Returns:
            None: Always returns None for pandas implementation.
        """
        return None

    @property
    def shape(self):
        """Return the dimensions of the DataFrame.

        Returns:
            tuple: (n_rows, n_columns) tuple.
        """
        return self.data.shape

    @property
    def labels_dict(self):
        return self._labels_dict

    def _get_column_index(
        self, column_name: Sequence[str] | str
    ) -> int | Sequence[int]:
        """Get integer position(s) of column(s) by name(s).

        Args:
            column_name: Single column name (str) or list of column names.

        Returns:
            int or Sequence[int]: Column position(s).

        Raises:
            ValueError: If column_name type is not str or list.
        """
        if isinstance(column_name, str):
            return self.data.columns.get_loc(column_name)
        elif isinstance(column_name, list):
            return self.data.columns.get_indexer(column_name)
        else:
            raise ValueError("Wrong column_name type.")

    def get_column_type(
        self, column_name: Iterable[str] | str = None
    ) -> dict[str, type] | type | None:
        """Get Python type(s) corresponding to pandas dtype(s) of column(s).

        Maps pandas dtypes to native Python types:
            - integer -> int
            - float -> float
            - bool -> bool
            - string/object/category -> str
            - list-like objects -> object

        Args:
            column_name: Column name(s). If None, returns types for all columns.

        Returns:
            type or Dict[str, type]: Single type if column_name is str,
            otherwise dict mapping column names to types.
        """
        column_name = self.data.columns if column_name is None else column_name
        dtypes = {}
        for k, v in self.data[column_name].dtypes.items():
            if pd.api.types.is_integer_dtype(v):
                dtypes[k] = int
            elif pd.api.types.is_float_dtype(v):
                dtypes[k] = float
            elif pd.api.types.is_object_dtype(v) and pd.api.types.is_list_like(
                self.data[column_name].iloc[0]
            ):
                dtypes[k] = object
            elif (
                pd.api.types.is_string_dtype(v)
                or pd.api.types.is_object_dtype(v)
                or v == "category"
            ):
                dtypes[k] = str
            elif pd.api.types.is_bool_dtype(v):
                dtypes[k] = bool
        if isinstance(column_name, Iterable):
            return dtypes
        else:
            if column_name in dtypes:
                return dtypes[column_name]
        return None

    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> pd.DataFrame:
        """Cast DataFrame columns to specified dtypes.

        Args:
            dtype: Dict mapping column names to target Python types.
            errors: Error handling: 'raise' (default) or 'ignore'.

        Returns:
            pd.DataFrame: DataFrame with casted columns.
        """
        return self.data.astype(dtype=dtype, errors=errors)

    def update_column_type(self, dtype: dict[str, type]):
        """Update column types, skipping columns with NaN values.

        Only updates columns that have no missing values to avoid
        type conversion errors.

        Args:
            dtype: Dict mapping column names to target Python types.

        Returns:
            self: Returns self for method chaining.
        """
        for column_name, type_name in dtype.items():
            if not self.data[column_name].isna().any():
                # if isinstance(type_name, str):
                #     self.data = self.data.replace({column_name : self.labels_dict[column_name]})
                #     self.labels_dict.pop(column_name)
                self.data = self.astype({column_name: type_name})
        return self

    def add_column(
        self,
        data: Any,
        name: str | list[str],
        index: Sequence | None = None,
    ):
        """Add a new column to the DataFrame.

        Args:
             Sequence of values for the new column.
            name: Column name (str) or single-element list containing name.
            index: Optional index labels for the new column. If None,
                uses existing DataFrame index.

        Returns:
            None: Modifies self.data in-place.
        """
        if isinstance(name, list) and len(name) == 1:
            name = name[0]
        if isinstance(data, pd.DataFrame):
            data = data.values
        else:
            data = Adapter.to_list(data)
        if len(self.data) != len(data):
            if isinstance(data[0], Iterable) and len(data[0]) == 1:
                data = data.squeeze()
            data = pd.Series(data)
        if index:
            self.data = self.data.join(
                pd.DataFrame(data, columns=[name], index=list(index))
            )
        else:
            self.data.loc[:, name] = data

    def append(self, other, reset_index: bool = False, axis: int = 0) -> pd.DataFrame:
        """Append other PandasDataset(s) to current DataFrame.

        Args:
            other: Single PandasDataset or list of PandasDataset instances.
            reset_index: If True, reset index in result (default: False).
            axis: Axis to concatenate along: 0 for rows, 1 for columns.

        Returns:
            pd.DataFrame: Concatenated DataFrame.
        """
        other = Adapter.to_list(other)
        new_data = pd.concat([self.data] + [d.data for d in other], axis=axis)
        if reset_index:
            new_data = new_data.reset_index(drop=True)
        return new_data

    def add_index_col(self, index_col_name: str | None = UTILITY_INDEX_COL_NAME):
        self.data[index_col_name] = np.array(range(len(self.data)))

    def remove_index_col(self, index_col_name: str | None = UTILITY_INDEX_COL_NAME):
        self.data = self.data.drop(index_col_name)

    def from_dict(self, data: FromDictTypes, index: Iterable | Sized | None = None):
        """Load data from dict-like structure into DataFrame.

        Args:
             Dict or list of dicts to convert to DataFrame.
            index: Optional index to assign to resulting DataFrame.

        Returns:
            self: Returns self for method chaining.
        """
        if isinstance(data, dict):
            self.data = pd.DataFrame().from_records(data, columns=list(data.keys()))
        else:
            self.data = pd.DataFrame().from_records(data)
        if index is not None:
            self.data.index = index
        return self

    def to_dict(self) -> dict[str, Any]:
        # """Convert DataFrame to dict with 'data' and 'index' keys.

        # Returns:
        #     dict: Format {"data": {col: [values]}, "index": [index_values]}.
        # """
        # return {
        #     "data": {
        #         column: self.data[column].to_list() for column in self.data.columns
        #     },
        #     "index": list(self.index),
        # }
        return self.data.to_dict()

    def to_records(self) -> list[dict]:
        """Convert DataFrame to list of row-wise dictionaries.

        Returns:
            list[dict]: Each dict represents a row with column names as keys.
        """
        return self.data.to_dict(orient="records")

    def loc(self, items: Iterable) -> Iterable:
        """Label-based selection wrapper ensuring DataFrame return type.

        Args:
            items: Labels or boolean array for selection.

        Returns:
            pd.DataFrame: Selected data as DataFrame.
        """
        data = self.data.loc[items]
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def iloc(self, items: Iterable) -> Iterable:
        """Integer-position-based selection wrapper ensuring DataFrame return type.

        Args:
            items: Integer positions or slices for selection.

        Returns:
            pd.DataFrame: Selected data as DataFrame.
        """
        data = self.data.iloc[items]
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


class PandasDataset(PandasNavigation, DatasetBackendCalc):
    @staticmethod
    def _convert_agg_result(result):
        """Convert aggregation results to appropriate return type.

        Handles edge cases:
            - pd.Series -> pd.DataFrame
            - Single-value (1x1) DataFrame -> float
            - Otherwise returns pd.DataFrame

        Args:
            result: Output from pandas aggregation methods.

        Returns:
            float or pd.DataFrame: Simplified result for single values,
            otherwise DataFrame.
        """
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if result.shape == (1, 1):
            return float(result.loc[result.index[0], result.columns[0]])
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def __init__(
        self,
        data: pd.DataFrame | dict | str | pd.Series | None = None,
        data_compression: str | None = None,
        non_compresion_cols: list | None = None,
    ):
        super().__init__(data, data_compression, non_compresion_cols)

    def get(self, key, default=None) -> Any:
        """Get value for key from underlying DataFrame, with default fallback.

        Args:
            key: Column name or DataFrame key.
            default: Value to return if key not found.

        Returns:
            Column data or default value.
        """
        return self.data.get(key, default)

    def take(
        self,
        indices: int | list[int],
        axis: Literal["index", "columns", "rows"] | int = 0,
    ) -> Any:
        """Return elements at specified positions along axis.

        Args:
            indices: Integer or list of integer positions.
            axis: Axis to select from: 0/'index'/'rows' for rows,
                1/'columns' for columns.

        Returns:
            Selected data from DataFrame.
        """
        return self.data.take(indices=indices, axis=axis)

    def apply(self, func: Callable, **kwargs) -> pd.DataFrame:
        """Apply function along axis of DataFrame.

        Args:
            func: Function to apply to each column/row.
            **kwargs: Additional arguments, including 'column_name' to name
                result column if output is Series.

        Returns:
            pd.DataFrame: Result of apply operation.
        """
        single_column_name = kwargs.pop("column_name")
        result = self.data.apply(func, **kwargs)
        if not isinstance(result, pd.DataFrame):
            result = result.to_frame(name=single_column_name)
        return self._wrap_result(result)

    def map(self, func: Callable, **kwargs) -> pd.DataFrame:
        """Map values using element-wise function.

        Args:
            func: Function to apply element-wise.
            **kwargs: Additional arguments passed to pandas map().

        Returns:
            pd.DataFrame: Mapped values.
        """
        return self._wrap_result(self.data.map(func, **kwargs))

    def is_empty(self) -> bool:
        """Check if DataFrame is empty (no rows or columns).

        Returns:
            bool: True if DataFrame is empty, False otherwise.
        """
        return self.data.empty

    def unique(self):
        """Get unique values for each column.

        Returns:
            dict: Mapping of column names to arrays of unique values.
        """
        return {column: self.data[column].unique() for column in self.data.columns}

    def nunique(self, dropna: bool = True):
        """Count number of unique values per column.

        Args:
            dropna: If True (default), exclude NaN from counts.

        Returns:
            dict: Mapping of column names to unique value counts.
        """
        return {column: self.data[column].nunique() for column in self.data.columns}

    def groupby(self, by: str | Iterable[str], **kwargs) -> pd.Grouper:
        """Group DataFrame by specified column(s).

        Args:
            by: Column name(s) to group by.
            **kwargs: Additional arguments passed to pandas groupby().

        Returns:
            pd.Grouper
        """
        return self.data.groupby(by=by, observed=False, **kwargs)

    def iter_groups(self, by: list[str]):
        for key, group in self.data.groupby(by=by, observed=False):
            yield key, group

    def grouped_value_counts(self, by: list[str], feature_cols: list[str]):
        result = {
            col: self.data.groupby(by=by, observed=False)[col].value_counts().to_dict()
            for col in feature_cols
        }
        # result[col] is {(group_key, val): count}; pivot to {group_key: {val: count}}
        rows: dict = {}
        for col, vc_dict in result.items():
            for composite_key, count in vc_dict.items():
                group_key = (
                    composite_key[0] if len(by) == 1 else composite_key[: len(by)]
                )
                val = composite_key[-1]
                rows.setdefault(group_key, {}).setdefault(col, {})[val] = count

        return {
            "data": {col: [rows[k].get(col, {}) for k in rows] for col in feature_cols},
            "index": list(rows.keys()),
        }

    def agg(self, func: str | list, **kwargs) -> pd.DataFrame | float:
        """Aggregate DataFrame using specified function(s).

        Args:
            func: Aggregation function name(s) (e.g., 'sum', 'mean') or list.
            **kwargs: Additional arguments passed to pandas agg().

        Returns:
            float or pd.DataFrame: Aggregated result (simplified to float
            for single-value results).
        """
        func = func if isinstance(func, (list, dict)) else [func]
        result = self.data.agg(func, **kwargs)
        return self._convert_agg_result(result)

    def max(self) -> pd.DataFrame | float:
        """Return maximum value(s) of DataFrame.

        Returns:
            float or pd.DataFrame: Maximum value(s).
        """
        return self._wrap_result(self.agg(["max"]))

    def idxmax(self) -> pd.DataFrame | float:
        """Return index label(s) of maximum value(s).

        Returns:
            float or pd.DataFrame: Index label(s) of maximum value(s).
        """
        return self._wrap_result(self.agg(["idxmax"]))

    def min(self) -> pd.DataFrame | float:
        """Return minimum value(s) of DataFrame.

        Returns:
            float or pd.DataFrame: Minimum value(s).
        """
        return self._wrap_result(self.agg(["min"]))

    def count(self) -> pd.DataFrame | float:
        """Count non-NA values for each column.

        Returns:
            float or pd.DataFrame: Count of non-NA values.
        """
        return self._wrap_result(self.agg(["count"]))

    def sum(self) -> pd.DataFrame | float:
        """Return sum of values.

        Returns:
            float or pd.DataFrame: Sum of values.
        """
        return self._wrap_result(self.agg(["sum"]))

    def mean(self) -> pd.DataFrame | float:
        """Return mean of values.

        Returns:
            float or pd.DataFrame: Mean of values.
        """
        return self._wrap_result(self.agg(["mean"]))

    def mode(
        self, numeric_only: bool = False, dropna: bool = True
    ) -> pd.DataFrame | float:
        """Return mode(s) of values.

        Args:
            numeric_only: If True, only include numeric columns.
            dropna: If True (default), exclude NaN from results.

        Returns:
            pd.DataFrame: Mode values (may have multiple rows if multimodal).
        """
        return self._wrap_result(
            self.data.mode(numeric_only=numeric_only, dropna=dropna)
        )

    def var(
        self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False
    ) -> pd.DataFrame | float:
        """Return unbiased variance over requested axis.

        Args:
            skipna: Exclude NA/null values if True (default).
            ddof: Delta Degrees of Freedom (default: 1 for sample variance).
            numeric_only: Include only numeric columns if True.

        Returns:
            float or pd.DataFrame: Variance value(s).
        """
        return self._wrap_result(
            self.agg(["var"], skipna=skipna, ddof=ddof, numeric_only=numeric_only)
        )

    def log(self) -> pd.DataFrame:
        """Compute natural logarithm of all numeric values.

        Returns:
            pd.DataFrame: DataFrame with log-transformed values.
        """
        np_data = np.log(self.data.to_numpy())
        return self._wrap_result(pd.DataFrame(np_data, columns=self.data.columns))

    def std(self, skipna: bool = True, ddof: int = 1) -> pd.DataFrame | float:
        """Return sample standard deviation.

        Args:
            skipna: Exclude NA/null values if True (default).
            ddof: Delta Degrees of Freedom (default: 1).

        Returns:
            float or pd.DataFrame: Standard deviation value(s).
        """
        return self._wrap_result(self.agg(["std"], skipna=skipna, ddof=ddof))

    def cov(self):
        """Compute pairwise covariance of columns, excluding NA/null values.

        Returns:
            pd.DataFrame: Covariance matrix.
        """
        return self._wrap_result(self.data.cov(ddof=1))

    def quantile(self, q: float = 0.5) -> pd.DataFrame | float:
        """Return values at the given quantile.

        Args:
            q: Quantile(s) to compute (0 <= q <= 1). Default 0.5 (median).

        Returns:
            float or pd.DataFrame: Quantile value(s).
        """
        if isinstance(q, list) and len(q) > 1:
            return self.data.quantile(q=q)
        return self._wrap_result(self.agg(func="quantile", q=q))

    def coefficient_of_variation(self) -> pd.DataFrame | float:
        """Compute coefficient of variation (std/mean) for each column.

        Returns:
            float or pd.DataFrame: CV value(s). Returns scalar for single-column
            DataFrames.
        """
        data = (self.data.std() / self.data.mean()).to_frame().T
        data.index = ["cv"]
        if data.shape == (1, 1):
            return float(data.loc[data.index[0], data.columns[0]])
        return self._wrap_result(
            data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        )

    def sort_index(self, ascending: bool = True, **kwargs) -> pd.DataFrame:
        """Sort DataFrame by index labels.

        Args:
            ascending: Sort ascending if True (default), descending if False.
            **kwargs: Additional arguments passed to pandas sort_index().

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
        return self._wrap_result(self.data.sort_index(ascending=ascending, **kwargs))

    def get_numeric_columns(self) -> list[str]:
        """Get list of columns with numeric dtypes.

        Note: Uses ScalarType from utils to determine numeric types.

        Returns:
            list[str]: Names of numeric columns.
        """
        return self.df.select_dtypes(include=ScalarType).columns.tolist()

    def corr(
        self,
        numeric_only: bool = False,
    ) -> pd.DataFrame | float:
        """Compute pairwise correlation of columns using Pearson method.

        Args:
            numeric_only: Include only numeric columns if True.

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        return self._wrap_result(
            self.data.corr(method="pearson", numeric_only=numeric_only)
        )

    def isna(self) -> pd.DataFrame:
        """Detect missing values (NaN, None).

        Returns:
            pd.DataFrame: Boolean DataFrame indicating missing values.
        """
        return self._wrap_result(self.data.isna())

    def sort_values(
        self, by: str | list[str], ascending: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Sort DataFrame by values along specified column(s).

        Args:
            by: Column name(s) to sort by.
            ascending: Sort ascending if True (default).
            **kwargs: Additional arguments passed to pandas sort_values().

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
        return self._wrap_result(
            self.data.sort_values(by=by, ascending=ascending, **kwargs)
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Return frequency counts of unique values.

        Args:
            normalize: Return proportions instead of counts if True.
            sort: Sort results by value counts if True (default).
            ascending: Sort in ascending order if True.
            dropna: Exclude missing values if True (default).

        Returns:
            pd.DataFrame: DataFrame with value counts (reset index).
        """
        return self._wrap_result(
            self.data.value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            ).reset_index()
        )

    def fillna(
        self,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fill missing values using specified method or value.

        Args:
            values: Scalar or dict of column->value mappings to fill with.
            method: Fill method: 'ffill' (forward fill) or 'bfill' (backward fill).
            **kwargs: Additional arguments passed to pandas fillna/bfill/ffill.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.

        Raises:
            ValueError: If method is not 'bfill' or 'ffill'.
        """
        if method is not None:
            if method == "bfill":
                return self._wrap_result(self.data.bfill(**kwargs))
            elif method == "ffill":
                return self._wrap_result(self.data.ffill(**kwargs))
            else:
                raise ValueError(f"Wrong fill method: {method}")

        return self._wrap_result(self.data.fillna(value=values, **kwargs))

    def na_counts(self) -> pd.DataFrame | int:
        """Count missing values per column.

        Returns:
            int or pd.DataFrame: Count of NA values (scalar for single-column
            DataFrames, otherwise DataFrame with 'na_counts' index).
        """
        data = self.data.isna().sum().to_frame().T
        data.index = ["na_counts"]
        if data.shape[0] == 1 and data.shape[1] == 1:
            return int(data.loc[data.index[0], data.columns[0]])
        return self._wrap_result(
            data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        )

    def dot(self, other: PandasDataset | np.ndarray) -> pd.DataFrame:
        """Compute matrix multiplication with another DataFrame or array.

        Args:
            other: PandasDataset or numpy array to multiply with.

        Returns:
            pd.DataFrame: Result of matrix multiplication.
        """
        if isinstance(other, np.ndarray):
            other_df = pd.DataFrame(
                data=other,
                columns=self.columns if other.shape[1] == self.shape[1] else None,
            )
            result = self.data.dot(other_df.T)
            result.columns = (
                self.columns if other.shape[1] == self.shape[1] else result.columns
            )
        else:
            result = self.data.dot(other.data)
        return self._wrap_result(
            result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        )

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ) -> pd.DataFrame:
        """Remove missing values.

        Args:
            how: 'any' (default): drop if any NA; 'all': drop if all NA.
            subset: Column label(s) to consider for NA detection.
            axis: 0/'index'/'rows' to drop rows, 1/'columns' to drop columns.

        Returns:
            pd.DataFrame: DataFrame with missing values removed.
        """
        return self._wrap_result(self.data.dropna(how=how, subset=subset, axis=axis))

    def transpose(self, names: Sequence[str] | None = None) -> pd.DataFrame:
        """Transpose rows and columns.

        Args:
            names: Optional list of column names for transposed result.

        Returns:
            pd.DataFrame: Transposed DataFrame.
        """
        result = self.data.transpose()
        if names is not None:
            result.columns = names
        return self._wrap_result(
            result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        )

    def sample(
        self,
        frac: float | None = None,
        n: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Return random sample of items from DataFrame.

        Args:
            frac: Fraction of rows to return (0 <= frac <= 1).
            n: Number of rows to return (mutually exclusive with frac).
            random_state: Seed for random number generator for reproducibility.

        Returns:
            pd.DataFrame: Randomly sampled DataFrame.
        """
        return self._wrap_result(
            self.data.sample(n=n, frac=frac, random_state=random_state)
        )

    def select_dtypes(
        self,
        include: str | None = None,
        exclude: str | None = None,
    ) -> pd.DataFrame:
        """Select columns based on dtype.

        Args:
            include: Dtype(s) to include (e.g., 'number', 'object').
            exclude: Dtype(s) to exclude.

        Returns:
            pd.DataFrame: DataFrame with selected columns.
        """
        return self._wrap_result(
            self.data.select_dtypes(include=include, exclude=exclude)
        )

    def limit(self, num: int | None = None) -> Any:
        if not num:
            return self.data
        else:
            return self.data.iloc[:num]

    def isin(self, values: Iterable) -> pd.DataFrame:
        """Check if elements are contained in passed values.

        Args:
            values: Iterable of values to check against.

        Returns:
            pd.DataFrame: Boolean DataFrame indicating membership.
        """
        return self._wrap_result(self.data.isin(values))

    def merge(
        self,
        right: PandasDataset,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool | None = None,
        right_index: bool | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> pd.DataFrame:
        """Merge DataFrame with another PandasDataset.

        Args:
            right: PandasDataset to merge with.
            on: Column name(s) to join on (must exist in both).
            left_on: Column(s) from left DataFrame to join on.
            right_on: Column(s) from right DataFrame to join on.
            left_index: Use index from left DataFrame as join key.
            right_index: Use index from right DataFrame as join key.
            suffixes: Suffixes for overlapping column names.
            how: Join type: 'inner' (default), 'left', 'right', 'outer', 'cross'.

        Returns:
            pd.DataFrame: Merged DataFrame.

        Raises:
            MergeOnError: If specified join column(s) don't exist in datasets.
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

        return self._wrap_result(
            self.data.merge(
                right=right.data,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                suffixes=suffixes,
                how=how,
            )
        )

    def drop(
        self,
        labels: str | None = None,
        axis: int | None = None,
        columns: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Drop specified labels from rows or columns.

        Args:
            labels: Label(s) to drop.
            axis: 0 for rows, 1 for columns.
            columns: Column name(s) to drop (alternative to labels+axis).

        Returns:
            pd.DataFrame: DataFrame with specified labels removed.
        """
        return self._wrap_result(
            self.data.drop(labels=labels, axis=axis, columns=columns)
        )

    def filter(
        self,
        items: list | None = None,
        regex: str | None = None,
        column: str | None = None,
        axis: int = 0,
    ) -> pd.DataFrame:
        """Subset DataFrame using labels or regex matching.

        Args:
            items: List of labels to keep along specified axis.
            regex: Regular expression to match against labels.
            axis: 0 for row labels, 1 for column labels (default: 0).

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if (axis == 0) and (items is None) and (column is not None):
            return self.data[self.data[column]]
        return self._wrap_result(self.data.filter(items=items, regex=regex, axis=axis))

    def rename(self, columns: dict[str, str]) -> pd.DataFrame:
        """Rename columns using a mapping dictionary.

        Args:
            columns: Dict mapping old column names to new names.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        return self._wrap_result(self.data.rename(columns=columns))

    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> pd.DataFrame:
        """Replace values in DataFrame.

        Args:
            to_replace: Value(s) to replace. Can be scalar, list, dict, or DataFrame.
            value: Replacement value(s). Ignored if to_replace is dict.
            regex: If True, treat to_replace/value as regex patterns.

        Returns:
            pd.DataFrame: DataFrame with replaced values.
        """
        if isinstance(to_replace, pd.DataFrame) and len(to_replace.columns) == 1:
            to_replace = to_replace.iloc[:, 0]
        elif isinstance(to_replace, pd.Series):
            to_replace = to_replace.to_list()
        elif isinstance(to_replace, dict):
            return self._wrap_result(
                self.data.replace(to_replace=to_replace, regex=regex)
            )
        return self._wrap_result(
            self.data.replace(to_replace=to_replace, value=value, regex=regex)
        )

    def reindex(self, labels: str = "", fill_value: str | None = None) -> pd.DataFrame:
        """Conform DataFrame to new index with optional fill value.

        Args:
            labels: New index labels to conform to.
            fill_value: Value to use for missing labels (default: NaN).

        Returns:
            pd.DataFrame: Reindexed DataFrame.
        """
        return self._wrap_result(self.data.reindex(labels, fill_value=fill_value))

    def list_to_columns(self, column: str) -> pd.DataFrame:
        """Expand a column containing lists into multiple columns.

        Each element in the list becomes a separate column named
        '{column}_0', '{column}_1', etc.

        Args:
            column: Name of column containing list values.

        Returns:
            pd.DataFrame: DataFrame with expanded columns (or original if
            lists have length 1).
        """
        data = self.data
        n_cols = len(data.loc[0, column])

        data_expanded = (
            pd.DataFrame(
                data[column].to_list(), columns=[f"{column}_{i}" for i in range(n_cols)]
            )
            if n_cols > 1
            else data
        )

        return data_expanded

    def checkpoint(self):
        pass
