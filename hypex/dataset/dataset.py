"""
This module provides the core Dataset and ExperimentData classes for data manipulation and analysis.

The Dataset class extends DatasetBase to provide a rich interface for working with tabular data,
including support for various data operations, statistical functions, and role-based column management.

The ExperimentData class provides a container for organizing experimental data, analysis results,
and metadata.

Classes:
    Dataset: Main class for working with tabular data with role-based column management
    ExperimentData: Container class for organizing experimental data and results
    DatasetAdapter: Utility class for converting various data types to Dataset objects

Example:
    >>> roles = {'col1': InfoRole(), 'col2': InfoRole()}
    >>> data = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})
    >>> ds = Dataset(roles=roles, data=data)
    >>> ds.mean()
    col1    2.0
    col2    5.0
"""

import warnings
from copy import copy, deepcopy
from collections.abc import Iterable
from typing import (
    Union,
    List,
    Iterable,
    Any,
    Dict,
    Callable,
    Hashable,
    Optional,
    Tuple,
    Literal,
    Sequence,
)

import pandas as pd  # type: ignore

from ..utils import (
    ID_SPLIT_SYMBOL,
    BackendsEnum,
    ConcatBackendError,
    ConcatDataError,
    ExperimentDataEnum,
    FromDictTypes,
    MultiFieldKeyTypes,
    NotFoundInExperimentDataError,
    DataTypeError,
    BackendTypeError,
    ScalarType,
)
from .abstract import DatasetBase
from .roles import (
    StatisticRole,
    InfoRole,
    ABCRole,
    FilterRole,
    DefaultRole,
    AdditionalRole,
)
from ..utils.adapter import Adapter
from ..utils.errors import InvalidArgumentError


class Dataset(DatasetBase):
    """
    A class for working with tabular data that extends DatasetBase with additional functionality.

    The Dataset class provides a rich interface for data manipulation, statistical operations,
    and role-based column management. It supports multiple backends (currently pandas)
    and provides consistent access patterns regardless of the underlying data storage.

    Attributes:
        loc (Locker): Label-based indexer for accessing data
        iloc (ILocker): Integer-based indexer for accessing data
        roles (Dict[str, ABCRole]): Mapping of column names to their roles
        data (pd.DataFrame): The underlying data storage

    Args:
        roles (Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]]):
            Role definitions for columns
        data (Optional[Union[pd.DataFrame, str]]): Input data or path to data
        backend (Optional[BackendsEnum]): Backend storage type
        default_role (Optional[ABCRole]): Default role for columns without explicit roles
    """

    class Locker:
        """
        Label-based indexer for accessing data in the Dataset.

        Provides a pandas-like .loc[] interface for label-based indexing.
        """

        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item) -> "Dataset":
            """
            Get data subset using label-based indexing.
            """
            t_data = self.backend.loc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            """
            Set data values using label-based indexing.
            """
            column_name = item[1]
            column_data_type = self.roles[column_name].data_type
            if (
                column_data_type == None
                or (
                    isinstance(value, Iterable)
                    and all(isinstance(v, column_data_type) for v in value)
                )
                or isinstance(value, column_data_type)
            ):
                if column_name not in self.backend.data.columns:
                    raise KeyError("Column must be added by using add_column method.")
                else:
                    self.backend.data.loc[item] = value
            else:
                raise TypeError("Value type does not match the expected data type.")

    class ILocker:
        """
        Integer-based indexer for accessing data in the Dataset.

        Provides a pandas-like .iloc[] interface for integer-based indexing.
        """

        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item) -> "Dataset":
            """
            Get data subset using integer-based indexing.
            """
            t_data = self.backend.iloc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            """
            Set data values using integer-based indexing.
            """
            column_index = item[1]
            column_name = self.backend.data.columns[column_index]
            column_data_type = self.roles[column_name].data_type
            if (
                column_data_type == None
                or (
                    isinstance(value, Iterable)
                    and all(isinstance(v, column_data_type) for v in value)
                )  # check for backend specific list (?)
                or isinstance(value, column_data_type)
            ):
                if column_index >= len(self.backend.data.columns):
                    raise IndexError("Column must be added by using add_column method.")
                else:
                    self.backend.data.iloc[item] = value
            else:
                raise TypeError("Value type does not match the expected data type.")

    def __init__(
        self,
        roles: Union[
            Dict[ABCRole, Union[List[str], str]],
            Dict[str, ABCRole],
        ],
        data: Optional[Union[pd.DataFrame, str]] = None,
        backend: Optional[BackendsEnum] = None,
        default_role: Optional[ABCRole] = None,
    ):
        """
        Initialize a new Dataset instance.

        Args:
            roles: Role definitions for columns
            data: Input data or path to data
            backend: Backend storage type
            default_role: Default role for columns without explicit roles
        """
        super().__init__(roles, data, backend, default_role)
        self.loc = self.Locker(self._backend, self.roles)
        self.iloc = self.ILocker(self._backend, self.roles)

    def __getitem__(self, item: Union[Iterable, str, int]) -> "Dataset":
        """
        Get a subset of the dataset by column selection.

        Args:
            item: Column name(s) to select

        Returns:
            Dataset containing only the selected columns
        """
        if isinstance(item, Dataset):
            item = item.data
        items = (
            [item] if isinstance(item, str) or not isinstance(item, Iterable) else item
        )
        roles: Dict = {
            column: (
                self.roles[column]
                if column in self.columns and self.roles.get(column, False)
                else InfoRole()
            )
            for column in items
        }
        result = Dataset(data=self._backend.__getitem__(item), roles=roles)
        result.tmp_roles = {
            key: value for key, value in self.tmp_roles.items() if key in items
        }
        return result

    def __setitem__(self, key: str, value: Any):
        """
        Set values for a column in the dataset.

        Args:
            key: Column name
            value: Values to set

        Raises:
            TypeError: If value type doesn't match column's expected type
        """
        if isinstance(value, Dataset):
            value = value.data
        if key not in self.columns and isinstance(key, str):
            self.add_column(value, {key: InfoRole()})
            warnings.warn(
                "Column must be added by using add_column method.",
                category=SyntaxWarning,
            )
            self.data[key] = value
        else:
            column_data_type = self.roles[key].data_type
            if (
                column_data_type == None
                or (
                    isinstance(value, Iterable)
                    and all(isinstance(v, column_data_type) for v in value)
                )  # check for backend specific list (?)
                or isinstance(value, column_data_type)
            ):
                self.data[key] = value
            else:
                raise TypeError("Value type does not match the expected data type.")

    def __binary_magic_operator(self, other, func_name: str) -> Any:
        """
        Helper method for implementing binary operators.

        Args:
            other: Right-hand operand
            func_name: Name of operator function to call

        Returns:
            Result of binary operation

        Raises:
            DataTypeError: If other has invalid type
            BackendTypeError: If backends don't match
        """
        if not any(
            isinstance(other, t) for t in [Dataset, str, int, float, bool, Sequence]
        ):
            raise DataTypeError(type(other))
        func = getattr(self._backend, func_name)
        t_roles = deepcopy(self.roles)
        for role in t_roles.values():
            role.data_type = None
        if isinstance(other, Dataset):
            if type(other._backend) is not type(self._backend):
                raise BackendTypeError(type(other._backend), type(self._backend))
            other = other.rename(
                {
                    other.columns[i]: self.data.columns[i]
                    for i in range(len(other.columns))
                }
            ).backend
        return Dataset(roles=t_roles, data=func(other))

    # comparison operators:
    def __eq__(self, other):
        """
        Implement equality comparison.
        """
        return self.__binary_magic_operator(other=other, func_name="__eq__")

    def __ne__(self, other):
        """
        Implement inequality comparison.
        """
        return self.__binary_magic_operator(other=other, func_name="__ne__")

    def __le__(self, other):
        """
        Implement less than or equal comparison.
        """
        return self.__binary_magic_operator(other=other, func_name="__le__")

    def __lt__(self, other):
        """
        Implement less than comparison.
        """
        return self.__binary_magic_operator(other=other, func_name="__lt__")

    def __ge__(self, other):
        """
        Implement greater than or equal comparison.
        """
        return self.__binary_magic_operator(other=other, func_name="__ge__")

    def __gt__(self, other):
        """
        Implement greater than comparison.
        """
        return self.__binary_magic_operator(other=other, func_name="__gt__")

    # unary operators:
    def __pos__(self):
        """
        Implement unary positive.
        """
        return Dataset(roles=self.roles, data=(+self._backend))

    def __neg__(self):
        """
        Implement unary negation.
        """
        return Dataset(roles=self.roles, data=(-self._backend))

    def __abs__(self):
        """
        Implement absolute value.
        """
        return Dataset(roles=self.roles, data=abs(self._backend))

    def __invert__(self):
        """
        Implement bitwise inversion.
        """
        return Dataset(roles=self.roles, data=(~self._backend))

    def __round__(self, ndigits: int = 0):
        """
        Implement rounding.
        """
        return Dataset(roles=self.roles, data=round(self._backend, ndigits))

    def __bool__(self):
        """
        Implement truth value testing.
        """
        return not self._backend.is_empty()

    # Binary math operators:
    def __add__(self, other):
        """
        Implement addition.
        """
        return self.__binary_magic_operator(other=other, func_name="__add__")

    def __sub__(self, other):
        """
        Implement subtraction.
        """
        return self.__binary_magic_operator(other=other, func_name="__sub__")

    def __mul__(self, other):
        """
        Implement multiplication.
        """
        return self.__binary_magic_operator(other=other, func_name="__mul__")

    def __floordiv__(self, other):
        """
        Implement floor division.
        """
        return self.__binary_magic_operator(other=other, func_name="__floordiv__")

    def __div__(self, other):
        """
        Implement division.
        """
        return self.__binary_magic_operator(other=other, func_name="__div__")

    def __truediv__(self, other):
        """
        Implement true division.
        """
        return self.__binary_magic_operator(other=other, func_name="__truediv__")

    def __mod__(self, other):
        """
        Implement modulo.
        """
        return self.__binary_magic_operator(other=other, func_name="__mod__")

    def __pow__(self, other):
        """
        Implement exponentiation.
        """
        return self.__binary_magic_operator(other=other, func_name="__pow__")

    def __and__(self, other):
        """
        Implement bitwise and.
        """
        return self.__binary_magic_operator(other=other, func_name="__and__")

    def __or__(self, other):
        """
        Implement bitwise or.
        """
        return self.__binary_magic_operator(other=other, func_name="__or__")

    # Right math operators:
    def __radd__(self, other):
        """
        Implement reverse addition.
        """
        return self.__binary_magic_operator(other=other, func_name="__radd__")

    def __rsub__(self, other):
        """
        Implement reverse subtraction.
        """
        return self.__binary_magic_operator(other=other, func_name="__rsub__")

    def __rmul__(self, other):
        """
        Implement reverse multiplication.
        """
        return self.__binary_magic_operator(other=other, func_name="__rmul__")

    def __rfloordiv__(self, other):
        """
        Implement reverse floor division.
        """
        return self.__binary_magic_operator(other=other, func_name="__rfloordiv__")

    def __rdiv__(self, other):
        """
        Implement reverse division.
        """
        return self.__binary_magic_operator(other=other, func_name="__rdiv__")

    def __rtruediv__(self, other):
        """
        Implement reverse true division.
        """
        return self.__binary_magic_operator(other=other, func_name="__rtruediv__")

    def __rmod__(self, other):
        """
        Implement reverse modulo.
        """
        return self.__binary_magic_operator(other=other, func_name="__rmod__")

    def __rpow__(self, other) -> Any:
        """
        Implement reverse exponentiation.
        """
        return self.__binary_magic_operator(other=other, func_name="__rpow__")

    @property
    def index(self):
        """
        Get the index of the dataset.
        """
        return self.backend.index

    @index.setter
    def index(self, value):
        """
        Set the index of the dataset.
        """
        self.backend.data.index = value

    @property
    def data(self):
        """
        Get the underlying data.
        """
        return self._backend.data

    @data.setter
    def data(self, value):
        """
        Set the underlying data.
        """
        self.backend.data = value

    @property
    def columns(self):
        """
        Get the column names.
        """
        return self.backend.columns

    @staticmethod
    def create_empty(roles=None, index=None, backend=BackendsEnum.pandas) -> "Dataset":
        """
        Create an empty dataset.

        Args:
            roles: Role definitions for columns
            index: Index for the empty dataset
            backend: Backend storage type

        Returns:
            Empty Dataset instance
        """
        if roles is None:
            roles = {}
        index = [] if index is None else index
        columns = list(roles.keys())
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend.create_empty(index, columns)
        ds.data = ds.backend.data
        return ds

    def _convert_data_after_agg(self, result) -> Union["Dataset", float]:
        """
        Convert aggregation result to appropriate type.

        Args:
            result: Result of aggregation operation

        Returns:
            Dataset or float depending on result type
        """
        if isinstance(result, float):
            return result
        role: ABCRole = StatisticRole()
        return Dataset(data=result, roles={column: role for column in result.columns})

    def add_column(
        self,
        data,
        role: Optional[Dict[str, ABCRole]] = None,
        index: Optional[Iterable[Hashable]] = None,
    ):
        """
        Add a new column to the dataset.

        Args:
            data: Column data to add
            role: Role definition for the new column
            index: Index for the new column

        Returns:
            Self for method chaining

        Raises:
            ValueError: If role is None and data is not a Dataset
        """
        if role is None:
            if not isinstance(data, Dataset):
                raise ValueError("If role is None, data must be a Dataset")
            if any([col in self.columns for col in data.columns]):
                raise ValueError("Columns with the same name already exist")
            self.roles.update(data.roles)
            self._backend.add_column(
                data.data,
                data.columns,
                index,
            )
        else:
            if any([col in self.columns for col in role.keys()]):
                raise ValueError("Columns with the same name already exist")
            if isinstance(role, Dict) and any(
                [not isinstance(r, ABCRole) for r in role.values()]
            ):
                raise TypeError("Role values must be of type ABCRole")
            if isinstance(data, Dataset):
                data = data.data
            self.roles.update(role)
            self._backend.add_column(data, list(role.keys()), index)
        return self

    def _check_other_dataset(self, other):
        """
        Check compatibility with another dataset.

        Args:
            other: Dataset to check compatibility with

        Raises:
            ConcatDataError: If other is not a Dataset
            ConcatBackendError: If backends don't match
        """
        if not isinstance(other, Dataset):
            raise ConcatDataError(type(other))
        if type(other._backend) is not type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))

    def astype(
        self, dtype: Dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> "Dataset":
        """
        Change the data type of one or more columns.

        Parameters:
        - dtype: Dictionary where keys are column names and values are target types.
        - errors: If 'raise', raises an error on invalid types; if 'ignore', skips invalid types.

        Returns:
        - A new Dataset object with the specified data types applied.
        """

        for col, _ in dtype.items():
            if (errors == "raise") and (col not in self.columns):
                raise KeyError(f"Column '{col}' does not exist in the Dataset.")

        new_backend = deepcopy(self._backend)
        new_backend.data = new_backend.astype(dtype, errors)
        new_roles = deepcopy(self.roles)

        if errors == "ignore":
            for col, target_type in dtype.items():
                if new_backend.get_column_type(col) == target_type:
                    new_roles[col].data_type = target_type
        elif errors == "raise":
            for col, target_type in dtype.items():
                new_roles[col].data_type = target_type

        return Dataset(roles=new_roles, data=new_backend.data)

    def append(self, other, reset_index=False, axis=0) -> "Dataset":
        """
        Append rows or columns from another dataset.

        Args:
            other: Dataset to append
            reset_index: Whether to reset index after append
            axis: 0 for row append, 1 for column append

        Returns:
            New Dataset with appended data
        """
        other = Adapter.to_list(other)

        new_roles = deepcopy(self.roles)
        for o in other:
            self._check_other_dataset(o)
            new_roles.update(o.roles)

        return Dataset(
            roles=new_roles, data=self.backend.append(other, reset_index, axis)
        )

    # TODO: set backend by backend object
    @staticmethod
    def from_dict(
        data: FromDictTypes,
        roles: Union[
            Dict[ABCRole, Union[List[str], str]],
            Dict[str, ABCRole],
        ],
        backend: BackendsEnum = BackendsEnum.pandas,
        index=None,
    ) -> "Dataset":
        """
        Create a Dataset from a dictionary.

        Args:
            data: Dictionary containing data
            roles: Role definitions for columns
            backend: Backend storage type
            index: Index for the dataset

        Returns:
            New Dataset created from dictionary
        """
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend.from_dict(data, index)
        ds.data = ds._backend.data
        return ds

    # What is going to happen when a matrix is returned?
    def apply(
        self,
        func: Callable,
        role: Dict[str, ABCRole],
        axis: int = 0,
        **kwargs,
    ) -> "Dataset":
        """
        Apply a function to the dataset.

        Args:
            func: Function to apply
            role: Role definition for result
            axis: Axis to apply function along
            **kwargs: Additional arguments for func

        Returns:
            New Dataset with function applied
        """
        if self.is_empty():
            return deepcopy(self)
        tmp_data = self._backend.apply(
            func=func, axis=axis, column_name=list(role.keys())[0], **kwargs
        )
        tmp_roles = (
            {list(role.keys())[0]: list(role.values())[0]}
            if ((not tmp_data.any().any()) and len(role) > 1)
            else role
        )
        return Dataset(
            data=tmp_data,
            roles=tmp_roles,
        )

    def map(self, func, na_action=None, **kwargs) -> "Dataset":
        """
        Apply a mapping function to the dataset.

        Args:
            func: Mapping function to apply
            na_action: How to handle NA values
            **kwargs: Additional arguments for func

        Returns:
            New Dataset with mapping applied
        """
        return Dataset(
            roles=self.roles,
            data=self._backend.map(func=func, na_action=na_action, **kwargs),
        )

    def is_empty(self) -> bool:
        """
        Check if dataset is empty.
        """
        return self._backend.is_empty()

    def unique(self) -> Dict[str, List[Any]]:
        """
        Get unique values for each column.
        """
        return self._backend.unique()

    def nunique(self, dropna: bool = False) -> Dict[str, int]:
        """
        Count unique values for each column.

        Args:
            dropna: Whether to exclude NA values

        Returns:
            Dictionary mapping column names to counts
        """
        return self._backend.nunique(dropna)

    def isin(self, values: Iterable) -> "Dataset":
        """
        Check whether values are contained in the dataset.

        Args:
            values: Values to check for

        Returns:
            Boolean mask as Dataset
        """
        role: ABCRole = FilterRole()
        return Dataset(
            roles={column: role for column in self.roles.keys()},
            data=self._backend.isin(values),
        )

    def groupby(
        self,
        by: Any,
        func: Optional[Union[str, List]] = None,
        fields_list: Optional[Union[str, List]] = None,
        **kwargs,
    ) -> List[Tuple[str, "Dataset"]]:
        """
        Group dataset by values.

        Args:
            by: Column(s) to group by
            func: Aggregation function to apply
            fields_list: Columns to include in result
            **kwargs: Additional arguments for groupby

        Returns:
            List of (group_key, group_data) tuples
        """
        if isinstance(by, Dataset) and len(by.columns) == 1:
            self.data = self.data.reset_index(drop=True)
            datasets = [
                (group, Dataset(roles=self.roles, data=self.data.loc[group_data.index]))
                for group, group_data in by._backend.groupby(by=by.columns[0], **kwargs)
            ]
        else:
            datasets = [
                (group, Dataset(roles=self.roles, data=data))
                for group, data in self._backend.groupby(by=by, **kwargs)
            ]
        if fields_list:
            fields_list = Adapter.to_list(fields_list)
            datasets = [(i, data[fields_list]) for i, data in datasets]
        if func:
            datasets = [(i, data.agg(func)) for i, data in datasets]
        for dataset in datasets:
            if isinstance(dataset, Dataset):
                dataset[1].tmp_roles = self.tmp_roles
        return datasets

    def sort(
        self,
        by: Optional[MultiFieldKeyTypes] = None,
        ascending: bool = True,
        **kwargs,
    ):
        """
        Sort dataset by values.

        Args:
            by: Column(s) to sort by
            ascending: Sort order
            **kwargs: Additional arguments for sort

        Returns:
            Sorted Dataset
        """
        if by is None:
            return Dataset(
                roles=self.roles,
                data=self.backend.sort_index(ascending=ascending, **kwargs),
            )
        return Dataset(
            roles=self.roles,
            data=self.backend.sort_values(by=by, ascending=ascending, **kwargs),
        )

    def fillna(
        self,
        values: Union[ScalarType, Dict[str, ScalarType], None] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
        **kwargs,
    ):
        """
        Fill NA values.

        Args:
            values: Values to fill with
            method: Method for filling
            **kwargs: Additional arguments

        Returns:
            Dataset with NA values filled

        Raises:
            ValueError: If neither values nor method provided
        """
        if values is None and method is None:
            raise ValueError("Value or filling method must be provided")
        return Dataset(
            roles=self.roles,
            data=self.backend.fillna(values=values, method=method, **kwargs),
        )

    def mean(self):
        """
        Calculate mean of numeric columns.
        """
        return self._convert_data_after_agg(self._backend.mean())

    def max(self):
        """
        Calculate maximum of columns.
        """
        return self._convert_data_after_agg(self._backend.max())

    def reindex(self, labels, fill_value: Optional[Any] = None) -> "Dataset":
        """
        Reindex the dataset.

        Args:
            labels: New index labels
            fill_value: Value to use for missing values

        Returns:
            Reindexed Dataset
        """
        return Dataset(
            self.roles, data=self.backend.reindex(labels, fill_value=fill_value)
        )

    def idxmax(self):
        """
        Get index of maximum values.
        """
        return self._convert_data_after_agg(self._backend.idxmax())

    def min(self):
        """
        Calculate minimum of columns.
        """
        return self._convert_data_after_agg(self._backend.min())

    def count(self):
        """
        Count non-NA values.
        """
        if self.is_empty():
            return Dataset.create_empty({role: InfoRole() for role in self.roles})
        return self._convert_data_after_agg(self._backend.count())

    def sum(self):
        """
        Calculate sum of columns.
        """
        return self._convert_data_after_agg(self._backend.sum())

    def log(self):
        """
        Calculate natural logarithm.
        """
        return self._convert_data_after_agg(self._backend.log())

    def mode(self, numeric_only: bool = False, dropna: bool = True):
        """
        Calculate mode of columns.

        Args:
            numeric_only: Whether to include only numeric columns
            dropna: Whether to exclude NA values

        Returns:
            Mode values as Dataset
        """
        t_data = self._backend.mode(numeric_only=numeric_only, dropna=dropna)
        return Dataset(data=t_data, roles={role: InfoRole() for role in t_data.columns})

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False):
        """
        Calculate variance of columns.

        Args:
            skipna: Whether to exclude NA values
            ddof: Delta degrees of freedom
            numeric_only: Whether to include only numeric columns

        Returns:
            Variance values as Dataset
        """
        return self._convert_data_after_agg(
            self._backend.var(skipna=skipna, ddof=ddof, numeric_only=numeric_only)
        )

    def agg(self, func: Union[str, List]):
        """
        Aggregate using one or more operations.

        Args:
            func: Function or list of functions to apply

        Returns:
            Aggregated results as Dataset
        """
        return self._convert_data_after_agg(self._backend.agg(func))

    def std(self, skipna: bool = True, ddof: int = 1):
        """
        Calculate standard deviation.
        """
        return self._convert_data_after_agg(self._backend.std(skipna=skipna, ddof=ddof))

    def quantile(self, q: float = 0.5):
        """
        Calculate quantiles for each column.

        Args:
            q: Quantile to compute, between 0 and 1

        Returns:
            Quantile values for each column
        """
        return self._convert_data_after_agg(self._backend.quantile(q=q))

    def coefficient_of_variation(self):
        """
        Calculate coefficient of variation.
        """
        return self._convert_data_after_agg(self._backend.coefficient_of_variation())

    def corr(self, method="pearson", numeric_only=False):
        """
        Calculate correlation between columns.

        Args:
            method: Correlation method
            numeric_only: Whether to include only numeric columns

        Returns:
            Correlation matrix as Dataset
        """
        t_data = self._backend.corr(method=method, numeric_only=numeric_only)
        t_roles = {column: self.roles[column] for column in t_data.columns}
        return Dataset(roles=t_roles, data=t_data)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ):
        """
        Count unique values.

        Args:
            normalize: Return proportions instead of counts
            sort: Sort by counts
            ascending: Sort order
            dropna: Whether to exclude NA values

        Returns:
            Value counts as Dataset
        """
        t_data = self._backend.value_counts(
            normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
        )
        t_roles = deepcopy(self.roles)
        column_name = "proportion" if normalize else "count"
        if column_name not in t_data:
            t_data = t_data.rename(columns={0: column_name})
        t_roles[column_name] = StatisticRole()
        return Dataset(roles=t_roles, data=t_data)

    def na_counts(self):
        """Count NA values"""
        return self._convert_data_after_agg(self._backend.na_counts())

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: Union[str, Iterable[str], None] = None,
        axis: Union[Literal["index", "rows", "columns"], int] = 0,
    ):
        """
        Remove missing values from the dataset.

        Args:
            how: How to drop rows/columns with missing values.
                'any': Drop if any NA values are present
                'all': Drop only if all values are NA
            subset: Labels of columns to check for missing values, or None to use all columns
            axis: Which axis to drop values from
                0/'index'/'rows': Drop rows
                1/'columns': Drop columns

        Returns:
            Dataset: A new dataset with NA values removed according to specified criteria
        """
        # Drop NA values using backend implementation
        new_data = self._backend.dropna(how=how, subset=subset, axis=axis)

        # Update roles based on axis - keep all roles for row drops, filter for column drops
        new_roles = (
            self.roles
            if axis == 0
            else {column: self.roles[column] for column in new_data.columns}
        )

        # Return new dataset with updated data and roles
        return Dataset(
            roles=new_roles,
            data=new_data,
        )

    def isna(self):
        """
        Check for missing values in the dataset.

        Returns:
            Dataset: A boolean dataset indicating missing values
        """
        return self._convert_data_after_agg(self._backend.isna())

    def select_dtypes(self, include: Any = None, exclude: Any = None):
        """
        Select columns based on their dtype.

        Args:
            include: Dtypes to include
            exclude: Dtypes to exclude

        Returns:
            Dataset: Dataset with selected dtypes
        """
        # Filter data by dtypes
        t_data = self._backend.select_dtypes(include=include, exclude=exclude)

        # Keep only roles for remaining columns
        t_roles = {k: v for k, v in self.roles.items() if k in t_data.columns}
        return Dataset(roles=t_roles, data=t_data)

    def merge(
        self,
        right,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
    ):
        """
        Merge this dataset with another dataset.

        Args:
            right: Right dataset to merge with
            on: Column name(s) to join on
            left_on: Column(s) from left dataset to join on
            right_on: Column(s) from right dataset to join on
            left_index: Use left index as join key
            right_index: Use right index as join key
            suffixes: Suffixes to add to overlapping column names
            how: Type of merge to perform

        Returns:
            Dataset: Merged dataset

        Raises:
            DataTypeError: If right is not a Dataset
            BackendTypeError: If backends don't match
        """
        # Default to index merge if no columns specified
        if not any([on, left_on, right_on, left_index, right_index]):
            left_index = True
            right_index = True

        # Validate input types
        if not isinstance(right, Dataset):
            raise DataTypeError(type(right))
        if type(right._backend) is not type(self._backend):
            raise BackendTypeError(type(right._backend), type(self._backend))

        # Perform merge operation
        t_data = self._backend.merge(
            right=right._backend,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            how=how,
        )

        # Combine roles from both datasets
        t_roles = deepcopy(self.roles)
        t_roles.update(right.roles)

        # Handle suffixed column roles
        for c in t_data.columns:
            if f"{c}".endswith(suffixes[0]) and c[: -len(suffixes[0])] in self.columns:
                t_roles[c] = self.roles[c[: -len(suffixes[0])]]
            if f"{c}".endswith(suffixes[1]) and c[: -len(suffixes[1])] in right.columns:
                t_roles[c] = right.roles[c[: -len(suffixes[1])]]

        # Create final roles dict with only merged columns
        new_roles = {c: t_roles[c] for c in t_data.columns}
        return Dataset(roles=new_roles, data=t_data)

    def drop(self, labels: Any = None, axis: int = 1):
        """
        Drop specified labels from rows or columns.

        Args:
            labels: Labels to drop
            axis: 0 for rows, 1 for columns

        Returns:
            Dataset: Dataset with specified labels dropped
        """
        # Convert Dataset labels to list of indices
        if isinstance(labels, Dataset):
            labels = list(labels.index)

        # Drop specified labels
        t_data = self._backend.drop(labels=labels, axis=axis)

        # Update roles based on axis
        t_roles = (
            deepcopy(self.roles)
            if axis == 0
            else {c: self.roles[c] for c in t_data.columns}
        )
        return Dataset(roles=t_roles, data=t_data)

    def filter(
        self,
        items: Optional[List] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: Optional[int] = None,
    ) -> "Dataset":
        """
        Filter rows or columns using specified criteria.

        Args:
            items: List of items to include
            like: Include labels matching this substring
            regex: Include labels matching this regular expression
            axis: 0 for rows, 1 for columns

        Returns:
            Dataset: Filtered dataset
        """
        t_data = self._backend.filter(items=items, like=like, regex=regex, axis=axis)
        t_roles = {c: self.roles[c] for c in t_data.columns if c in self.roles.keys()}
        return Dataset(roles=t_roles, data=t_data)

    def dot(self, other: "Dataset") -> "Dataset":
        """
        Compute matrix multiplication with another dataset.

        Args:
            other: Dataset to multiply with

        Returns:
            Dataset: Result of matrix multiplication
        """
        return Dataset(roles=other.roles, data=self.backend.dot(other.backend))

    def transpose(
        self,
        roles: Optional[Union[Dict[str, ABCRole], List[str]]] = None,
    ) -> "Dataset":
        """
        Transpose the dataset.

        Args:
            roles: New roles for transposed columns

        Returns:
            Dataset: Transposed dataset
        """
        # Get role names if provided
        roles_names: List[Union[str, None]] = (
            list(roles.keys()) or [] if isinstance(roles, Dict) else roles
        )

        # Transpose data
        result_data = self.backend.transpose(roles_names)

        # Create default roles if none provided
        if roles is None or isinstance(roles, List):
            names = result_data.columns if roles is None else roles
            roles = {column: DefaultRole() for column in names}

        return Dataset(roles=roles, data=result_data)

    def sample(
        self,
        frac: Optional[float] = None,
        n: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> "Dataset":
        """
        Return a random sample of the dataset.

        Args:
            frac: Fraction of rows to sample
            n: Number of rows to sample
            random_state: Random seed for reproducibility

        Returns:
            Dataset: Random sample of rows
        """
        return Dataset(
            self.roles,
            data=self.backend.sample(frac=frac, n=n, random_state=random_state),
        )

    def cov(self):
        """
        Calculate covariance matrix.

        Returns:
            Dataset: Covariance matrix
        """
        t_data = self.backend.cov()
        return Dataset(
            {column: DefaultRole() for column in t_data.columns}, data=t_data
        )

    def rename(self, names: Dict[str, str]):
        """
        Rename columns.

        Args:
            names: Dictionary mapping old names to new names

        Returns:
            Dataset: Dataset with renamed columns
        """
        roles = {names.get(column, column): role for column, role in self.roles.items()}
        return Dataset(roles, data=self.backend.rename(names))

    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        regex: bool = False,
    ) -> "Dataset":
        """
        Replace values in the dataset.

        Args:
            to_replace: Values to replace
            value: Value to replace with
            regex: Whether to interpret to_replace as regular expressions

        Returns:
            Dataset: Dataset with replaced values
        """
        return Dataset(
            self.roles,
            data=self._backend.replace(to_replace=to_replace, value=value, regex=regex),
        )


class ExperimentData:
    """
    Class for managing experimental data and analysis results.
    """

    def __init__(self, data: Dataset):
        """
        Initialize ExperimentData.

        Args:
            data: Base dataset for the experiment
        """
        self._data = data
        self.additional_fields = Dataset.create_empty(index=data.index)
        self.variables: Dict[str, Dict[str, Union[int, float]]] = {}
        self.groups: Dict[str, Dict[str, Dataset]] = {}
        self.analysis_tables: Dict[str, Dataset] = {}
        self.id_name_mapping: Dict[str, str] = {}

    @property
    def ds(self):
        """
        Get the base dataset.
        """
        return self._data

    @staticmethod
    def create_empty(
        roles=None, backend=BackendsEnum.pandas, index=None
    ) -> "ExperimentData":
        """
        Create empty ExperimentData instance.

        Args:
            roles: Roles for columns
            backend: Backend to use
            index: Index for empty dataset

        Returns:
            ExperimentData: Empty experiment data instance
        """
        ds = Dataset.create_empty(backend, roles, index)
        return ExperimentData(ds)

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
        """
        Check if executor ID exists in specified space.

        Args:
            executor_id: ID to check
            space: Space to check in

        Returns:
            bool: Whether ID exists in space
        """
        if space == ExperimentDataEnum.additional_fields:
            return executor_id in self.additional_fields.columns
        elif space == ExperimentDataEnum.variables:
            return executor_id in self.variables.keys()
        elif space == ExperimentDataEnum.analysis_tables:
            return executor_id in self.analysis_tables
        else:
            return any(self.check_hash(executor_id, s) for s in ExperimentDataEnum)

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: Union[str, Dict[str, str]],
        value: Any,
        key: Optional[str] = None,
        role=None,
    ) -> "ExperimentData":
        """
        Set a value in the specified space.

        Args:
            space: Space to set value in
            executor_id: ID for the value
            value: Value to set
            key: Optional key for the value
            role: Optional role for the value

        Returns:
            ExperimentData: Self for chaining
        """
        # Handle additional fields
        if space == ExperimentDataEnum.additional_fields:
            if not isinstance(value, Dataset) or len(value.columns) == 1:
                self.additional_fields = self.additional_fields.add_column(
                    data=value, role={executor_id: role}
                )
            else:
                rename_dict = (
                    {value.columns[0]: executor_id}
                    if isinstance(executor_id, str)
                    else executor_id
                )
                value = value.rename(names=rename_dict)
                self.additional_fields = self.additional_fields.merge(
                    right=value, left_index=True, right_index=True
                )

        # Handle analysis tables
        elif space == ExperimentDataEnum.analysis_tables:
            self.analysis_tables[executor_id] = value

        # Handle variables
        elif space == ExperimentDataEnum.variables:
            if executor_id in self.variables:
                self.variables[executor_id][key] = value
            elif isinstance(value, Dict):
                self.variables[executor_id] = value
            else:
                self.variables[executor_id] = {key: value}

        # Handle groups
        elif space == ExperimentDataEnum.groups:
            if executor_id not in self.groups:
                self.groups[executor_id] = {key: value}
            else:
                self.groups[executor_id][key] = value

        return self

    def get_ids(
        self,
        classes: Union[type, Iterable[type], str, Iterable[str]],
        searched_space: Union[
            ExperimentDataEnum, Iterable[ExperimentDataEnum], None
        ] = None,
        key: Optional[str] = None,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get IDs matching specified criteria.

        Args:
            classes: Classes to search for
            searched_space: Spaces to search in
            key: Optional key to match

        Returns:
            Dict mapping classes to spaces and matching IDs
        """

        def check_id(id_: str, class_: str) -> bool:
            result = id_[: id_.find(ID_SPLIT_SYMBOL)] == class_

            if result and key is not None:
                result = id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :] == key
            return result

        # Define spaces to search
        spaces = {
            ExperimentDataEnum.additional_fields: self.additional_fields.columns,
            ExperimentDataEnum.analysis_tables: self.analysis_tables.keys(),
            ExperimentDataEnum.groups: self.groups.keys(),
            ExperimentDataEnum.variables: self.variables.keys(),
        }

        # Convert classes to names
        classes = [
            c.__name__ if isinstance(c, type) else c for c in Adapter.to_list(classes)
        ]

        # Get spaces to search
        searched_space = (
            Adapter.to_list(searched_space) if searched_space else list(spaces.keys())
        )

        # Return matching IDs
        return {
            class_: {
                space.value: [
                    str(id_) for id_ in spaces[space] if check_id(id_, class_)
                ]
                for space in searched_space
            }
            for class_ in classes
        }

    def get_one_id(
        self,
        class_: Union[type, str],
        space: ExperimentDataEnum,
        key: Optional[str] = None,
    ) -> str:
        """
        Get a single ID matching criteria.

        Args:
            class_: Class to search for
            space: Space to search in
            key: Optional key to match

        Returns:
            str: Matching ID

        Raises:
            NotFoundInExperimentDataError: If no matching ID found
        """
        class_ = class_ if isinstance(class_, str) else class_.__name__
        result = self.get_ids(class_, space, key)
        if (class_ not in result) or (not len(result[class_][space.value])):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

    def copy(self, data: Optional[Dataset] = None) -> "ExperimentData":
        """
        Create a deep copy.

        Args:
            data: Optional new base dataset

        Returns:
            ExperimentData: Deep copy of self
        """
        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result

    def field_search(
        self,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role: bool = False,
        search_types=None,
    ) -> List[str]:
        """
        Search for fields with specified roles.

        Args:
            roles: Roles to search for
            tmp_role: Whether to include temporary roles
            search_types: Types to search for

        Returns:
            List[str]: Matching field names
        """
        searched_field = []
        roles = Adapter.to_list(roles)

        # Split roles by type
        field_in_additional = [
            role for role in roles if isinstance(role, AdditionalRole)
        ]
        field_in_data = [role for role in roles if role not in field_in_additional]

        # Search in main data
        if field_in_data:
            searched_field += self.ds.search_columns(
                field_in_data, tmp_role=tmp_role, search_types=search_types
            )

        # Search in additional fields
        if field_in_additional and isinstance(self, ExperimentData):
            searched_field += self.additional_fields.search_columns(
                field_in_additional, tmp_role=tmp_role, search_types=search_types
            )

        return searched_field

    def field_data_search(
        self,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role: bool = False,
        search_types=None,
    ) -> Dataset:
        """
        Search for data with specified roles.

        Args:
            roles: Roles to search for
            tmp_role: Whether to include temporary roles
            search_types: Types to search for

        Returns:
            Dataset: Dataset containing matching fields
        """
        searched_data: Dataset = Dataset.create_empty()
        roles = Adapter.to_list(roles)

        # Map roles to columns
        roles_columns_map = {
            role: self.field_search(role, tmp_role, search_types) for role in roles
        }

        # Build dataset from matching columns
        for role, columns in roles_columns_map.items():
            for column in columns:
                t_data = (
                    self.additional_fields[column]
                    if isinstance(role, AdditionalRole)
                    else self.ds[column]
                )
                searched_data = searched_data.add_column(
                    data=t_data, role={column: role}
                )
        return searched_data


class DatasetAdapter(Adapter):
    """
    Adapter class for converting various data types to Dataset objects.
    """

    @staticmethod
    def to_dataset(
        data: Union[Dict, Dataset, pd.DataFrame, List, str, int, float, bool],
        roles: Union[ABCRole, Dict[str, ABCRole]],
    ) -> Dataset:
        """
        Convert various data types to a Dataset object.

        Args:
            data: Input data to convert
            roles: Roles for the dataset columns

        Returns:
            Dataset: Converted dataset

        Raises:
            InvalidArgumentError: If data type is not supported
        """
        # Convert data based on its type
        if isinstance(data, Dict):
            return DatasetAdapter.dict_to_dataset(data, roles)
        elif isinstance(data, pd.DataFrame):
            if isinstance(roles, ABCRole):
                raise InvalidArgumentError("roles", "Dict[str, ABCRole]")
            return DatasetAdapter.frame_to_dataset(data, roles)
        elif isinstance(data, List):
            if isinstance(roles, ABCRole):
                raise InvalidArgumentError("roles", "Dict[str, ABCRole]")
            return DatasetAdapter.list_to_dataset(data, roles)
        elif any(isinstance(data, t) for t in [str, int, float, bool]):
            return DatasetAdapter.value_to_dataset(data, roles)
        elif isinstance(data, Dataset):
            return data
        else:
            raise InvalidArgumentError("data", "Dict, pd.DataFrame, List, Dataset")

    @staticmethod
    def value_to_dataset(
        data: ScalarType, roles: Union[ABCRole, Dict[str, ABCRole]]
    ) -> Dataset:
        """
        Convert a scalar value to a Dataset.

        Args:
            data: Scalar value to convert
            roles: Roles for the dataset

        Returns:
            Dataset: Dataset containing the scalar value
        """
        if isinstance(roles, ABCRole):
            roles = {"value": roles}
        return Dataset(roles=roles, data=pd.DataFrame({list(roles.keys())[0]: [data]}))

    @staticmethod
    def dict_to_dataset(
        data: Dict, roles: Union[ABCRole, Dict[str, ABCRole]]
    ) -> Dataset:
        """
        Convert a dictionary to a Dataset.

        Args:
            data: Dictionary to convert
            roles: Roles for the dataset

        Returns:
            Dataset: Dataset created from dictionary
        """
        roles_names = list(data.keys())
        if any(
            [
                any(isinstance(i, t) for t in [int, str, float, bool])
                for i in list(data.values())
            ]
        ):
            data = [data]
        if isinstance(roles, Dict):
            return Dataset.from_dict(data=data, roles=roles)
        elif isinstance(roles, ABCRole):
            return Dataset.from_dict(
                data=data, roles={name: roles for name in roles_names}
            )

    @staticmethod
    def list_to_dataset(data: List, roles: Dict[str, ABCRole]) -> Dataset:
        """
        Convert a list to a Dataset.

        Args:
            data: List to convert
            roles: Roles for the dataset

        Returns:
            Dataset: Dataset created from list
        """
        return Dataset(
            roles=roles,
            data=pd.DataFrame(data=data, columns=[list(roles.keys())[0]]),
        )

    @staticmethod
    def frame_to_dataset(data: pd.DataFrame, roles: Dict[str, ABCRole]) -> Dataset:
        """
        Convert a pandas DataFrame to a Dataset.

        Args:
            data: DataFrame to convert
            roles: Roles for the dataset

        Returns:
            Dataset: Dataset created from DataFrame
        """
        return Dataset(
            roles=roles,
            data=data,
        )
