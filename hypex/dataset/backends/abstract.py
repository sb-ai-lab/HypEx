from __future__ import annotations

try:
    from typing import Self  # Python >= 3.11
except ImportError:
    from typing_extensions import Self  # Python < 3.11

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence, Sized
from typing import Any, Literal

from ...utils import AbstractMethodError, FromDictTypes


class DatasetBackendNavigation(ABC):
    """Abstract interface for dataset navigation and basic I/O.

    Subclasses must implement the abstract methods to provide backend-specific
    behaviour (e.g., pandas, PySpark).  The class defines the minimal set of
    operations required for data inspection, indexing, type handling, and
    structural modifications.
    """

    @property
    def name(self) -> str:
        """Return a canonical name derived from the class name."""
        return str(self.__class__.__name__).lower().replace("backend", "")

    @property
    @abstractmethod
    def index(self):
        """Return the index (row labels) of the underlying data structure.

        Returns:
            The index object appropriate for the backend (e.g., pd.Index,
            list, etc.).
        """
        raise AbstractMethodError

    @property
    @abstractmethod
    def columns(self):
        """Return the column labels of the underlying data structure.

        Returns:
            The column labels appropriate for the backend.
        """
        raise AbstractMethodError

    @abstractmethod
    def from_dict(self, data: FromDictTypes, index: Iterable | Sized | None = None):
        """Build/replace the internal data from a dict-like structure.

        Args:
            data: A dictionary or list of records that can be converted to a
                tabular structure.
            index: Optional index to assign.  Must be convertible to the
                backend's native index type.

        Returns:
            Self, enabling method chaining.
        """
        raise AbstractMethodError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Export the dataset as a dictionary.

        Returns:
            A dict with keys ``'data'`` (column‑oriented dict of value
            lists) and ``'index'`` (list of index values).
        """
        raise AbstractMethodError

    def to_records(self) -> list[dict]:
        """Convert the dataset to a list of row‑wise dictionaries.

        Returns:
            A list of dicts, each representing one row with column names
            as keys.
        """
        raise AbstractMethodError

    @abstractmethod
    def __getitem__(self, item: Any) -> Any:
        """Enable indexing/slicing access to the data.

        Args:
            item: An indexer (int, slice, str, list of str, etc.).

        Returns:
            The selected subset of data.  The exact type depends on the
            backend and the nature of ``item`` (e.g., a scalar, Series, or
            a new DatasetBackendNavigation instance).
        """
        raise AbstractMethodError

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        raise AbstractMethodError

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------
    @abstractmethod
    def __eq__(self, other: Any) -> Self:
        """Element‑wise equality comparison (``self == other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __ne__(self, other: Any) -> Self:
        """Element‑wise inequality comparison (``self != other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __le__(self, other: Any) -> Self:
        """Element‑wise less‑than‑or‑equal comparison (``self <= other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __lt__(self, other: Any) -> Self:
        """Element‑wise less‑than comparison (``self < other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __ge__(self, other: Any) -> Self:
        """Element‑wise greater‑than‑or‑equal comparison (``self >= other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __gt__(self, other: Any) -> Self:
        """Element‑wise greater‑than comparison (``self > other``)."""
        raise AbstractMethodError

    # ------------------------------------------------------------------
    # Unary arithmetic operators
    # ------------------------------------------------------------------
    @abstractmethod
    def __pos__(self) -> Self:
        """Unary positive (``+self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __neg__(self) -> Self:
        """Unary negation (``-self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __abs__(self) -> Self:
        """Element‑wise absolute value (``abs(self)``)."""
        raise AbstractMethodError

    @abstractmethod
    def __invert__(self) -> Self:
        """Bitwise/logical NOT (``~self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __round__(self, ndigits: int = 0) -> Self:
        """Round to ``ndigits`` decimal places.

        Args:
            ndigits: Number of decimal places (default 0).
        """
        raise AbstractMethodError

    # ------------------------------------------------------------------
    # Binary arithmetic operators
    # ------------------------------------------------------------------
    @abstractmethod
    def __add__(self, other: Any) -> Self:
        """Element‑wise addition (``self + other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __sub__(self, other: Any) -> Self:
        """Element‑wise subtraction (``self - other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __mul__(self, other) -> Self:
        """Element‑wise multiplication (``self * other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __floordiv__(self, other: Any) -> Self:
        """Element‑wise floor division (``self // other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __div__(self, other) -> Self:
        """Element‑wise division (``self / other``) – legacy Python 2 style."""
        raise AbstractMethodError

    @abstractmethod
    def __truediv__(self, other: Any) -> Self:
        """Element‑wise true division (``self / other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __mod__(self, other: Any) -> Self:
        """Element‑wise modulo (``self % other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __pow__(self, other: Any) -> Self:
        """Element‑wise exponentiation (``self ** other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __and__(self, other: Any) -> Self:
        """Element‑wise bitwise/logical AND (``self & other``)."""
        raise AbstractMethodError

    @abstractmethod
    def __or__(self, other: Any) -> Self:
        """Element‑wise bitwise/logical OR (``self | other``)."""
        raise AbstractMethodError

    # ------------------------------------------------------------------
    # Reflected (right) arithmetic operators
    # ------------------------------------------------------------------
    @abstractmethod
    def __radd__(self, other: Any) -> Self:
        """Reflected addition (``other + self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __rsub__(self, other: Any) -> Self:
        """Reflected subtraction (``other - self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __rmul__(self, other: Any) -> Self:
        """Reflected multiplication (``other * self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __rfloordiv__(self, other: Any) -> Self:
        """Reflected floor division (``other // self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __rdiv__(self, other: Any) -> Self:
        """Reflected division (``other / self``) – legacy Python 2 style."""
        raise AbstractMethodError

    @abstractmethod
    def __rtruediv__(self, other: Any) -> Self:
        """Reflected true division (``other / self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __rmod__(self, other: Any) -> Self:
        """Reflected modulo (``other % self``)."""
        raise AbstractMethodError

    @abstractmethod
    def __rpow__(self, other: Any) -> Self:
        """Reflected exponentiation (``other ** self``)."""
        raise AbstractMethodError

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        raise AbstractMethodError

    @abstractmethod
    def _repr_html_(self) -> str:
        """Return an HTML representation (for Jupyter/IPython)."""
        raise AbstractMethodError

    # ------------------------------------------------------------------
    # Structural operations
    # ------------------------------------------------------------------
    @abstractmethod
    def create_empty(
        self,
        index: Iterable | None = None,
        columns: Iterable[str] | None = None,
    ) -> Self:
        """Create an empty dataset with the specified index and columns.

        Args:
            index:  Optional iterable of labels for the index.
            columns:  Optional iterable of column names.

        Returns:
            Self, enabling chaining (often a new instance is returned).
        """
        raise AbstractMethodError

    @abstractmethod
    def _get_column_index(
        self, column_name: Sequence[str] | str
    ) -> int | Sequence[int]:
        """Convert column name(s) to integer position(s).

        Args:
            column_name: A single column name (str) or a sequence of names.

        Returns:
            For a single name – its integer position.  For a sequence –
            a sequence of integer positions (e.g., list).
        """
        raise AbstractMethodError

    @abstractmethod
    def get_column_type(
        self, column_name: str | Iterable[str] | None = None
    ) -> dict[str, type] | type | None:
        """Infer Python type(s) from the dtype(s) of the column(s).

        Args:
            column_name:
                * ``str`` – return the type for that column.
                * ``Iterable[str]`` – return a dict mapping each column to its type.
                * ``None`` – return a dict mapping **all** columns to their types.

        Returns:
            The type or a mapping of column names to types, or ``None`` if the
            column does not exist.
        """
        raise AbstractMethodError

    @abstractmethod
    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> Self:
        """Cast columns to the specified types.

        Args:
            dtype: Mapping of column names to target Python types.
            errors: If ``'raise'``, raise on failure; ``'ignore'`` suppresses
                errors.

        Returns:
            Self with updated column types.
        """
        raise AbstractMethodError

    @abstractmethod
    def update_column_type(self, dtype: dict[str, type]) -> Self:
        """Update column types, skipping columns that contain null values.

        Args:
            dtype: Mapping of column names to target Python types.

        Returns:
            Self, enabling method chaining.
        """
        raise AbstractMethodError

    @abstractmethod
    def add_column(
        self, data: Any, name: str | None = None, index: Sequence | None = None
    ) -> None:
        """Add a new column to the dataset (in‑place).

        Args:
            data: Values for the new column (list, Series, array‑like).
            name: Column name(s).  A single‑element list is automatically
                unpacked to a string.
            index: Optional index for the column.  If ``None``, the existing
                dataset index is used.
        """
        raise AbstractMethodError

    @abstractmethod
    def append(self, other: Any, reset_index: bool = False, axis: int = 0) -> Self:
        """Concatenate ``other`` dataset(s) to the current one.

        Args:
            other: A single dataset or an iterable of datasets (all must be
                compatible with the backend).
            reset_index: If ``True``, reset the index in the resulting dataset.
            axis: Axis along which to concatenate – ``0`` for rows, ``1`` for
                columns.

        Returns:
            A new dataset containing the combined data.
        """
        raise AbstractMethodError

    @abstractmethod
    def loc(self, values: Iterable) -> Self:
        """Label‑based selection.

        Args:
            values: Row labels (or boolean mask, etc.) accepted by the
                backend's ``loc`` indexer.

        Returns:
            A dataset containing the selected rows.
        """
        raise AbstractMethodError

    @abstractmethod
    def iloc(self, values: Iterable) -> Self:
        """Integer‑position‑based selection.

        Args:
            values: Integer positions (or slices) accepted by the backend's
                ``iloc`` indexer.

        Returns:
            A dataset containing the selected rows.
        """
        raise AbstractMethodError


class DatasetBackendCalc(DatasetBackendNavigation, ABC):
    """Abstract interface extending navigation with calculation/analysis methods.

    Subclasses must implement statistical summaries, transformations,
    grouping, merging, and other high‑level operations.
    """

    @abstractmethod
    def mean(self) -> Any:
        """Compute mean of numeric columns.

        Returns:
            A scalar or a dataset of means (implementation‑dependent).
        """
        raise AbstractMethodError

    @abstractmethod
    def mode(self, numeric_only: bool = False, dropna: bool = True) -> Any:
        """Compute the mode(s) of each column.

        Args:
            numeric_only: If True, consider only numeric columns.
            dropna: If True, exclude missing values from the result.

        Returns:
            A dataset with the mode value(s).  May contain multiple rows if
            multiple values share the highest frequency.
        """
        raise AbstractMethodError

    @abstractmethod
    def var(
        self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False
    ) -> Any:
        """Compute unbiased variance.

        Args:
            skipna: Exclude NA/null values when computing.
            ddof: Delta degrees of freedom (divisor is ``N - ddof``).
            numeric_only: Include only numeric columns.

        Returns:
            Scalar or dataset of variances.
        """
        raise AbstractMethodError

    @abstractmethod
    def max(self) -> Any:
        """Return maximum value(s)."""
        raise AbstractMethodError

    @abstractmethod
    def idxmax(self) -> Any:
        """Return index label(s) of the maximum value(s)."""
        raise AbstractMethodError

    @abstractmethod
    def min(self) -> Any:
        """Return minimum value(s)."""
        raise AbstractMethodError

    @abstractmethod
    def count(self) -> Any:
        """Count non‑NA values for each column."""
        raise AbstractMethodError

    @abstractmethod
    def sum(self) -> Any:
        """Return sum of values."""
        raise AbstractMethodError

    @abstractmethod
    def log(self) -> Any:
        """Compute natural logarithm of all numeric values."""
        raise AbstractMethodError

    @abstractmethod
    def agg(self, func: str | list | dict) -> Any:
        """Aggregate using one or more functions.

        Args:
            func: Aggregation function name(s) or a dict mapping column
                names to function(s).

        Returns:
            Aggregated result – a scalar if a single value is produced by
            all functions, otherwise a dataset.
        """
        raise AbstractMethodError

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` (column name) or ``default`` if not found."""
        raise AbstractMethodError

    @abstractmethod
    def take(
        self,
        indices: int | list[int],
        axis: Literal["index", "columns", "rows"] | int = 0,
    ) -> Any:
        """Return elements at given integer positions along an axis.

        Args:
            indices: Position(s) of elements to select.
            axis: ``0`` / ``'index'`` / ``'rows'`` for rows,
                ``1`` / ``'columns'`` for columns.

        Returns:
            Selected subset.
        """
        raise AbstractMethodError

    @abstractmethod
    def get_values(self, row: str | None = None, column: str | None = None) -> Any:
        """Retrieve values by label.

        Args:
            row: A row label. If ``None``, all rows are selected.
            column: A column name. If ``None``, all columns are selected.

        Returns:
            The selected data (scalar, list, or dataset).
        """
        raise AbstractMethodError

    @abstractmethod
    def iget_values(self, row: int | None = None, column: int | None = None) -> Any:
        """Retrieve values by integer position (iloc style).

        Args:
            row: Integer row position. If ``None``, all rows.
            column: Integer column position. If ``None``, all columns.

        Returns:
            The selected data.
        """
        raise AbstractMethodError

    @abstractmethod
    def apply(self, func: Callable, **kwargs) -> Any:
        """Apply a function along an axis of the data.

        Args:
            func: Function to apply to each column (or row).
            **kwargs: Additional keyword arguments forwarded to the backend
                (e.g., ``column_name`` for naming the result).

        Returns:
            Transformed dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def map(self, func: Callable, **kwargs) -> Any:
        """Apply a function element‑wise.

        Args:
            func: Function to apply to every element.
            **kwargs: Additional arguments for the backend's ``map``.

        Returns:
            Dataset with mapped values.
        """
        raise AbstractMethodError

    @abstractmethod
    def is_empty(self) -> bool:
        """Return ``True`` if the dataset contains no rows or columns."""
        raise AbstractMethodError

    @abstractmethod
    def unique(self) -> Any:
        """Return unique values for each column.

        Returns:
            Typically a dict mapping column names to arrays/lists of unique
            values.
        """
        raise AbstractMethodError

    @abstractmethod
    def nunique(self, dropna: bool = True) -> Any:
        """Count number of distinct values per column.

        Args:
            dropna: If True, exclude missing values.

        Returns:
            A dict mapping column names to counts.
        """
        raise AbstractMethodError

    @abstractmethod
    def isin(self, values: Iterable) -> Any:
        """Test whether each element is contained in ``values``.

        Args:
            values: Iterable of values to check against.

        Returns:
            Boolean dataset of the same shape.
        """
        raise AbstractMethodError

    @abstractmethod
    def groupby(self, by: str | Iterable[str], **kwargs) -> Any:
        """Group the dataset by specified column(s).

        Args:
            by: Column name(s) defining the groups.
            **kwargs: Additional arguments for the backend's ``groupby``.

        Returns:
            A group‑by object that supports aggregation.
        """
        raise AbstractMethodError

    @abstractmethod
    def sort_index(self, **kwargs) -> Any:
        """Sort by index labels.

        Args:
            **kwargs: Arguments forwarded to the backend's ``sort_index``
                (e.g., ``ascending``).

        Returns:
            Sorted dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def sort_values(self, by: str | list[str], ascending: bool = True, **kwargs) -> Any:
        """Sort by values in specified column(s).

        Args:
            by: Column name(s).
            ascending: Sort order.
            **kwargs: Additional arguments for the backend.

        Returns:
            Sorted dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def std(self, skipna: bool = True, ddof: int = 1) -> Any:
        """Return sample standard deviation.

        Args:
            skipna: Exclude NA/null values.
            ddof: Delta degrees of freedom (default 1).

        Returns:
            Scalar or dataset of standard deviations.
        """
        raise AbstractMethodError

    @abstractmethod
    def coefficient_of_variation(self) -> Any:
        """Compute coefficient of variation (std / mean) for numeric columns.

        Returns:
            Scalar or dataset containing CV values.  Division by zero
            yields NaN.
        """
        raise AbstractMethodError

    @abstractmethod
    def get_numeric_columns(self) -> list[str]:
        """Return list of column names whose dtype is numeric."""
        raise AbstractMethodError

    @abstractmethod
    def corr(self, numeric_only: bool = False) -> Any:
        """Compute pairwise Pearson correlation of numeric columns.

        Args:
            numeric_only: If True, only include numeric columns.

        Returns:
            Correlation matrix as a dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Any:
        """Return frequency counts of unique values (computed on the first
        column by default).

        Args:
            normalize: If True, return proportions.
            sort: Sort by frequency.
            ascending: Sort ascending.
            dropna: Exclude NA values.

        Returns:
            A dataset with columns (value, count) plus an index column.
        """
        raise AbstractMethodError

    @abstractmethod
    def grouped_value_counts(self, by: list[str], feature_cols: list[str]) -> Any:
        """Return a dataframe indexed by group keys where each cell is a
        ``{category: count}`` dict for the corresponding feature column.

        Args:
            by: List of columns defining the groups.
            feature_cols: List of columns for which to compute
                category counts.

        Returns:
            A dict with keys ``'data'`` (a dict of column‑oriented lists
            of dicts) and ``'index'`` (list of group keys).
        """
        raise AbstractMethodError

    @abstractmethod
    def na_counts(self) -> Any:
        """Count missing values per column.

        Returns:
            A dataset with a single row (index ``'na_counts'``) showing
            the count of NA values in each column, or an integer for a
            single‑column dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ) -> Any:
        """Remove missing values.

        Args:
            how: ``'any'`` drops if any NA present, ``'all'`` drops only
                if all values are NA.
            subset: Column label(s) to consider.
            axis: ``0`` / ``'index'`` / ``'rows'`` to drop rows,
                ``1`` / ``'columns'`` to drop columns.

        Returns:
            Dataset with NA values removed.
        """
        raise AbstractMethodError

    @abstractmethod
    def isna(self) -> Any:
        """Detect missing values.

        Returns:
            Boolean dataset indicating where NA values occur.
        """
        raise AbstractMethodError

    @abstractmethod
    def quantile(self, q: float = 0.5) -> Any:
        """Return values at the given quantile.

        Args:
            q: Quantile(s) to compute (0 <= q <= 1). Default 0.5 (median).

        Returns:
            Scalar or dataset of quantile values.
        """
        raise AbstractMethodError

    @abstractmethod
    def select_dtypes(
        self, include: Any | None = None, exclude: Any | None = None
    ) -> Any:
        """Select columns based on dtype.

        Args:
            include: Dtype(s) to include.
            exclude: Dtype(s) to exclude.

        Returns:
            Dataset with the selected columns.
        """
        raise AbstractMethodError

    @abstractmethod
    def limit(self, num: int | None = None) -> Any:
        """Limit the number of rows.

        Args:
            num: Maximum number of rows.  If None, all rows are returned.

        Returns:
            A dataset with at most ``num`` rows.
        """
        raise AbstractMethodError

    @abstractmethod
    def merge(
        self,
        right: Any,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> Any:
        """Merge with another dataset using database‑style joins.

        Args:
            right: The dataset to merge with.
            on: Column name(s) to join on (must exist in both datasets).
            left_on: Columns from the left dataset to use as keys.
            right_on: Columns from the right dataset to use as keys.
            left_index: Use the index of the left dataset as join key.
            right_index: Use the index of the right dataset as join key.
            suffixes: Suffixes to append to overlapping column names.
            how: Join type.

        Returns:
            Merged dataset.

        Raises:
            MergeOnError: If specified join keys are not found.
        """
        raise AbstractMethodError

    @abstractmethod
    def drop(
        self,
        labels: str | None = None,
        axis: int | None = None,
        columns: str | Iterable[str] | None = None,
    ) -> Any:
        """Drop specified labels from rows or columns.

        Args:
            labels: Label(s) to drop (alternative to ``columns``).
            axis: 0 for rows, 1 for columns.
            columns: Column name(s) to drop.

        Returns:
            Dataset without the dropped elements.
        """
        raise AbstractMethodError

    @abstractmethod
    def filter(
        self,
        items: list | None = None,
        regex: str | None = None,
        column: str | None = None,
        axis: int = 0,
    ) -> Any:
        """Subset rows or columns based on label or regex.

        Args:
            items: List of labels to keep.
            regex: Regular expression to match labels.
            column: If provided and ``axis=0``, use this column for
                boolean filtering (``self.data[column]``).
            axis: 0 for rows, 1 for columns.

        Returns:
            Filtered dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def fillna(
        self,
        values: Any = None,
        method: Literal["bfill", "ffill"] | None = None,
        **kwargs,
    ) -> Any:
        """Fill missing values.

        Args:
            values: Scalar or dict of column‑wise values.
            method: Fill method – ``'bfill'`` or ``'ffill'``.
            **kwargs: Additional arguments passed to the backend's fillna.

        Returns:
            Dataset with missing values filled.

        Raises:
            ValueError: If an unknown method is supplied.
        """
        raise AbstractMethodError

    @abstractmethod
    def dot(self, other: Any) -> Any:
        """Compute matrix multiplication with ``other``.

        Args:
            other: A dataset, numpy array, or compatible object.

        Returns:
            Result of the dot product as a dataset or scalar.
        """
        raise AbstractMethodError

    @abstractmethod
    def transpose(self, names: Sequence[str] | None = None) -> Any:
        """Transpose rows and columns.

        Args:
            names: Optional list of column names for the transposed result.

        Returns:
            Transposed dataset.
        """
        raise AbstractMethodError

    @abstractmethod
    def rename(self, columns: dict[str, str]) -> Any:
        """Rename columns using a mapping dictionary.

        Args:
            columns: ``{old_name: new_name}`` mapping.

        Returns:
            Dataset with renamed columns.
        """
        raise AbstractMethodError

    @abstractmethod
    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> Any:
        """Replace values.

        Args:
            to_replace: Value(s) to be replaced.
            value: Replacement value(s).
            regex: Treat ``to_replace`` and ``value`` as regex patterns.

        Returns:
            Dataset with replaced values.
        """
        raise AbstractMethodError

    @abstractmethod
    def checkpoint(self) -> None:
        """Perform a backend‑specific checkpoint (e.g., materialise in Spark).

        Raises:
            NotImplementedError: If not supported by the backend.
        """
        raise AbstractMethodError
