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
    """A class for working with tabular data that extends DatasetBase with additional functionality.

    The Dataset class provides a rich interface for data manipulation, statistical operations,
    and role-based column management. It supports multiple backends (currently pandas)
    and provides consistent access patterns regardless of the underlying data storage.

    Args:
        roles (Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]]): 
            Role definitions for columns. Can be either:
            - Dict mapping role objects to column names
            - Dict mapping column names to role objects
        data (Optional[Union[pd.DataFrame, str]], optional): Input data or path to data. Defaults to None.
        backend (Optional[BackendsEnum], optional): Backend storage type. Defaults to None.
        default_role (Optional[ABCRole], optional): Default role for columns without explicit roles. Defaults to None.

    Attributes:
        loc (Locker): Label-based indexer for accessing data
        iloc (ILocker): Integer-based indexer for accessing data
        roles (Dict[str, ABCRole]): Mapping of column names to their roles
        data (pd.DataFrame): The underlying data storage

    Examples:
        Create a dataset with roles and data:
        >>> import pandas as pd
        >>> from hypex.dataset import Dataset
        >>> from hypex.dataset.roles import InfoRole, StatisticRole
        >>> 
        >>> df = pd.DataFrame({
        ...     'id': [1, 2, 3],
        ...     'value': [10.0, 20.0, 30.0]
        ... })
        >>> roles = {
        ...     'id': InfoRole(),
        ...     'value': StatisticRole()
        ... }
        >>> ds = Dataset(roles=roles, data=df)

        Access data using loc and iloc:
        >>> ds.loc[0:1, 'value']  # Label-based access
        >>> ds.iloc[0:1, 1]  # Integer-based access

        Perform operations:
        >>> ds['value'] * 2  # Column multiplication
        >>> ds[ds['value'] > 20]  # Filtering
    """

    class Locker:
        """Label-based indexer for accessing data in the Dataset.

        Provides a pandas-like .loc[] interface for label-based indexing.

        Args:
            backend: The backend data storage object
            roles: Dictionary mapping column names to roles

        Examples:
            >>> ds = Dataset(roles=roles, data=df)
            >>> # Get rows 0-1 for 'value' column
            >>> ds.loc[0:1, 'value']
            >>> # Set values for specific labels
            >>> ds.loc[0:1, 'value'] = [100.0, 200.0]
        """

        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item) -> "Dataset":
            """Get data subset using label-based indexing.

            Args:
                item: Label-based index specification

            Returns:
                Dataset: A new Dataset containing the selected data
            """
            t_data = self.backend.loc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            """Set data values using label-based indexing.

            Args:
                item: Label-based index specification
                value: Values to set

            Raises:
                KeyError: If trying to set values for a non-existent column
                TypeError: If value type does not match the column's expected data type
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
        """Integer-based indexer for accessing data in the Dataset.

        Provides a pandas-like .iloc[] interface for integer-based indexing.

        Args:
            backend: The backend data storage object
            roles: Dictionary mapping column names to roles

        Examples:
            >>> ds = Dataset(roles=roles, data=df)
            >>> # Get first row, second column
            >>> ds.iloc[0, 1]
            >>> # Set values using integer indexing
            >>> ds.iloc[0:2, 1] = [100.0, 200.0]
        """

        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item) -> "Dataset":
            """Get data subset using integer-based indexing.

            Args:
                item: Integer-based index specification

            Returns:
                Dataset: A new Dataset containing the selected data
            """
            t_data = self.backend.iloc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            """Set data values using integer-based indexing.

            Args:
                item: Integer-based index specification
                value: Values to set

            Raises:
                IndexError: If trying to set values for a non-existent column index
                TypeError: If value type does not match the column's expected data type
            """
            column_index = item[1]
            column_name = self.backend.data.columns[column_index]
            column_data_type = self.roles[column_name].data_type
            if (
                column_data_type == None
                or (
                    isinstance(value, Iterable)
                    and all(isinstance(v, column_data_type) for v in value)
                )
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
        """Initialize a new Dataset instance.

        Args:
            roles: Role definitions for columns
            data: Input data or path to data
            backend: Backend storage type
            default_role: Default role for columns without explicit roles

        Examples:
            >>> from hypex.dataset import Dataset
            >>> from hypex.dataset.roles import InfoRole
            >>> 
            >>> # Create with DataFrame
            >>> df = pd.DataFrame({'id': [1, 2, 3]})
            >>> ds = Dataset(roles={'id': InfoRole()}, data=df)
            >>> 
            >>> # Create with CSV path
            >>> ds = Dataset(
            ...     roles={'id': InfoRole()},
            ...     data='data.csv'
            ... )
        """
        super().__init__(roles, data, backend, default_role)
        self.loc = self.Locker(self._backend, self.roles)
        self.iloc = self.ILocker(self._backend, self.roles)

    def __getitem__(self, item: Union[Iterable, str, int]) -> "Dataset":
        """Get a subset of the dataset by column selection.

        Args:
            item (Union[Iterable, str, int]): Column name(s) to select. Can be:
                - String: Single column name
                - Iterable: Multiple column names
                - Integer: Column position

        Returns:
            Dataset: A new Dataset containing only the selected columns

        Examples:
            >>> ds = Dataset(roles=roles, data=df)
            >>> # Select single column
            >>> ds['value']
            >>> # Select multiple columns
            >>> ds[['id', 'value']]
            >>> # Select by position
            >>> ds[0]
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
        """Set values for a column in the dataset.

        Args:
            key (str): Column name
            value (Any): Values to set. Must match the column's data type if specified.

        Raises:
            TypeError: If value type doesn't match column's expected type
            SyntaxWarning: If adding a new column (use add_column() instead)

        Examples:
            >>> ds = Dataset(roles=roles, data=df)
            >>> # Update existing column
            >>> ds['value'] = [40.0, 50.0, 60.0]
            >>> # Set from another dataset
            >>> ds['value'] = other_ds['value']
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
                )
                or isinstance(value, column_data_type)
            ):
                self.data[key] = value
            else:
                raise TypeError("Value type does not match the expected data type.")

    def __binary_magic_operator(self, other, func_name: str) -> Any:
        """Helper method for implementing binary operators.

        Args:
            other: Right-hand operand
            func_name (str): Name of operator function to call

        Returns:
            Result of binary operation

        Raises:
            DataTypeError: If other has invalid type
            BackendTypeError: If backends don't match

        Examples:
            >>> ds = Dataset(roles=roles, data=df)
            >>> # Add scalar
            >>> ds['value'] + 10
            >>> # Multiply datasets
            >>> ds['value'] * other_ds['value']
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
        """Implement equality comparison.

        Args:
            other: Value to compare with

        Returns:
            Dataset: Boolean mask of equality comparison

        Examples:
            >>> ds['value'] == 20.0
            >>> ds['value'] == other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__eq__")

    def __ne__(self, other):
        """Implement inequality comparison.

        Args:
            other: Value to compare with

        Returns:
            Dataset: Boolean mask of inequality comparison

        Examples:
            >>> ds['value'] != 20.0
            >>> ds['value'] != other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__ne__")

    def __le__(self, other):
        """Implement less than or equal comparison.

        Args:
            other: Value to compare with

        Returns:
            Dataset: Boolean mask of less than or equal comparison

        Examples:
            >>> ds['value'] <= 20.0
            >>> ds['value'] <= other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__le__")

    def __lt__(self, other):
        """Implement less than comparison.

        Args:
            other: Value to compare with

        Returns:
            Dataset: Boolean mask of less than comparison

        Examples:
            >>> ds['value'] < 20.0
            >>> ds['value'] < other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__lt__")

    def __ge__(self, other):
        """Implement greater than or equal comparison.

        Args:
            other: Value to compare with

        Returns:
            Dataset: Boolean mask of greater than or equal comparison

        Examples:
            >>> ds['value'] >= 20.0
            >>> ds['value'] >= other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__ge__")

    def __gt__(self, other):
        """Implement greater than comparison.

        Args:
            other: Value to compare with

        Returns:
            Dataset: Boolean mask of greater than comparison

        Examples:
            >>> ds['value'] > 20.0
            >>> ds['value'] > other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__gt__")

    # unary operators:
    def __pos__(self):
        """Implement unary positive.

        Returns:
            Dataset: Dataset with positive values

        Examples:
            >>> +ds['value']
        """
        return Dataset(roles=self.roles, data=(+self._backend))

    def __neg__(self):
        """Implement unary negation.

        Returns:
            Dataset: Dataset with negated values

        Examples:
            >>> -ds['value']
        """
        return Dataset(roles=self.roles, data=(-self._backend))

    def __abs__(self):
        """Implement absolute value.

        Returns:
            Dataset: Dataset with absolute values

        Examples:
            >>> abs(ds['value'])
        """
        return Dataset(roles=self.roles, data=abs(self._backend))

    def __invert__(self):
        """Implement bitwise inversion.

        Returns:
            Dataset: Dataset with inverted values

        Examples:
            >>> ~ds['boolean_column']
        """
        return Dataset(roles=self.roles, data=(~self._backend))

    def __round__(self, ndigits: int = 0):
        """Implement rounding.

        Args:
            ndigits (int, optional): Number of decimal places. Defaults to 0.

        Returns:
            Dataset: Dataset with rounded values

        Examples:
            >>> round(ds['value'])
            >>> round(ds['value'], 2)
        """
        return Dataset(roles=self.roles, data=round(self._backend, ndigits))

    def __bool__(self):
        """Implement truth value testing.

        Returns:
            bool: True if dataset is not empty, False otherwise

        Examples:
            >>> bool(ds)  # True if dataset has data
        """
        return not self._backend.is_empty()

    # Binary math operators:
    def __add__(self, other):
        """Implement addition.

        Args:
            other: Value to add

        Returns:
            Dataset: Result of addition

        Examples:
            >>> ds['value'] + 10
            >>> ds['value'] + other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__add__")

    def __sub__(self, other):
        """Implement subtraction.

        Args:
            other: Value to subtract

        Returns:
            Dataset: Result of subtraction

        Examples:
            >>> ds['value'] - 10
            >>> ds['value'] - other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__sub__")

    def __mul__(self, other):
        """Implement multiplication.

        Args:
            other: Value to multiply by

        Returns:
            Dataset: Result of multiplication

        Examples:
            >>> ds['value'] * 2
            >>> ds['value'] * other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__mul__")

    def __floordiv__(self, other):
        """Implement floor division.

        Args:
            other: Value to divide by

        Returns:
            Dataset: Result of floor division

        Examples:
            >>> ds['value'] // 2
            >>> ds['value'] // other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__floordiv__")

    def __div__(self, other):
        """Implement division.

        Args:
            other: Value to divide by

        Returns:
            Dataset: Result of division

        Examples:
            >>> ds['value'] / 2
            >>> ds['value'] / other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__div__")

    def __truediv__(self, other):
        """Implement true division.

        Args:
            other: Value to divide by

        Returns:
            Dataset: Result of true division

        Examples:
            >>> ds['value'] / 2
            >>> ds['value'] / other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__truediv__")

    def __mod__(self, other):
        """Implement modulo.

        Args:
            other: Value to take modulo with

        Returns:
            Dataset: Result of modulo operation

        Examples:
            >>> ds['value'] % 3
            >>> ds['value'] % other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__mod__")

    def __pow__(self, other):
        """Implement exponentiation.

        Args:
            other: Value to raise to power of

        Returns:
            Dataset: Result of exponentiation

        Examples:
            >>> ds['value'] ** 2
            >>> ds['value'] ** other_ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__pow__")

    def __and__(self, other):
        """Implement bitwise and.

        Args:
            other: Value to AND with

        Returns:
            Dataset: Result of bitwise AND

        Examples:
            >>> ds['boolean_col'] & True
            >>> ds['boolean_col'] & other_ds['boolean_col']
        """
        return self.__binary_magic_operator(other=other, func_name="__and__")

    def __or__(self, other):
        """Implement bitwise or.

        Args:
            other: Value to OR with

        Returns:
            Dataset: Result of bitwise OR

        Examples:
            >>> ds['boolean_col'] | True
            >>> ds['boolean_col'] | other_ds['boolean_col']
        """
        return self.__binary_magic_operator(other=other, func_name="__or__")

    # Right math operators:
    def __radd__(self, other):
        """Implement reverse addition.

        Args:
            other: Value to add

        Returns:
            Dataset: Result of addition

        Examples:
            >>> 10 + ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__radd__")

    def __rsub__(self, other):
        """Implement reverse subtraction.

        Args:
            other: Value to subtract from

        Returns:
            Dataset: Result of subtraction

        Examples:
            >>> 10 - ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rsub__")

    def __rmul__(self, other):
        """Implement reverse multiplication.

        Args:
            other: Value to multiply by

        Returns:
            Dataset: Result of multiplication

        Examples:
            >>> 2 * ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rmul__")

    def __rfloordiv__(self, other):
        """Implement reverse floor division.

        Args:
            other: Value to divide from

        Returns:
            Dataset: Result of floor division

        Examples:
            >>> 10 // ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rfloordiv__")

    def __rdiv__(self, other):
        """Implement reverse division.

        Args:
            other: Value to divide from

        Returns:
            Dataset: Result of division

        Examples:
            >>> 10 / ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rdiv__")

    def __rtruediv__(self, other):
        """Implement reverse true division.

        Args:
            other: Value to divide from

        Returns:
            Dataset: Result of true division

        Examples:
            >>> 10 / ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rtruediv__")

    def __rmod__(self, other):
        """Implement reverse modulo.

        Args:
            other: Value to take modulo from

        Returns:
            Dataset: Result of modulo operation

        Examples:
            >>> 10 % ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rmod__")

    def __rpow__(self, other) -> Any:
        """Implement reverse exponentiation.

        Args:
            other: Base value

        Returns:
            Dataset: Result of exponentiation

        Examples:
            >>> 2 ** ds['value']
        """
        return self.__binary_magic_operator(other=other, func_name="__rpow__")

    @property
    def index(self):
        """Get the index of the dataset.

        Returns:
            Index: The dataset's index

        Examples:
            >>> ds.index
            >>> ds.index = pd.Index(['a', 'b', 'c'])
        """
        return self.backend.index

    @index.setter
    def index(self, value):
        """Set the index of the dataset.

        Args:
            value: New index to set
        """
        self.backend.data.index = value

    @property
    def data(self):
        """Get the underlying data.

        Returns:
            pd.DataFrame: The underlying pandas DataFrame

        Examples:
            >>> ds.data
            >>> ds.data = new_df
        """
        return self._backend.data

    @data.setter
    def data(self, value):
        """Set the underlying data.

        Args:
            value: New data to set
        """
        self.backend.data = value

    @property
    def columns(self):
        """Get the column names.

        Returns:
            Index: The column names of the dataset

        Examples:
            >>> ds.columns
        """
        return self.backend.columns
    @staticmethod
    def create_empty(roles=None, index=None, backend=BackendsEnum.pandas) -> "Dataset":
        """Creates an empty dataset with specified roles and index.

        Args:
            roles (Optional[Dict[str, ABCRole]]): Role definitions for columns. Maps column names to role objects.
                Defaults to empty dict.
            index (Optional[List]): Index values for the empty dataset. Defaults to empty list.
            backend (BackendsEnum): Backend storage type to use. Defaults to pandas.

        Returns:
            Dataset: A new empty Dataset instance with the specified roles and index.

        Examples:
            Create empty dataset with no roles or index:
            >>> ds = Dataset.create_empty()

            Create with roles and custom index:
            >>> from hypex.dataset.roles import InfoRole, StatisticRole
            >>> roles = {
            ...     'id': InfoRole(),
            ...     'value': StatisticRole()
            ... }
            >>> ds = Dataset.create_empty(
            ...     roles=roles,
            ...     index=['a', 'b', 'c']
            ... )
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
        """Converts aggregation result to appropriate type.

        Internal method to handle conversion of aggregation results to either a Dataset
        or float value.

        Args:
            result (Union[pd.DataFrame, float]): Result from an aggregation operation.

        Returns:
            Union[Dataset, float]: Either:
                - A float if the result was a single numeric value
                - A new Dataset with StatisticRole applied to all columns

        Examples:
            Internal usage:
            >>> ds = Dataset(...)
            >>> agg_result = ds._backend.mean()
            >>> converted = ds._convert_data_after_agg(agg_result)
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
        """Adds a new column or columns to the dataset.

        Args:
            data (Union[Dataset, pd.Series, pd.DataFrame]): Column data to add. Can be:
                - A Dataset instance
                - A pandas Series/DataFrame
                - Other array-like data
            role (Optional[Dict[str, ABCRole]]): Role definitions for new columns.
                Maps column names to role objects. Required if data is not a Dataset.
            index (Optional[Iterable[Hashable]]): Index for the new column(s).
                Must match length of data.

        Returns:
            Dataset: Self for method chaining.

        Raises:
            ValueError: If:
                - role is None and data is not a Dataset
                - Column names already exist in dataset
            TypeError: If role values are not ABCRole instances

        Examples:
            Add single column with role:
            >>> from hypex.dataset.roles import StatisticRole
            >>> ds = Dataset(...)
            >>> ds.add_column(
            ...     data=[1, 2, 3],
            ...     role={'new_col': StatisticRole()}
            ... )

            Add columns from another dataset:
            >>> other_ds = Dataset(...)
            >>> ds.add_column(other_ds)

            Add with custom index:
            >>> ds.add_column(
            ...     data=[1, 2, 3],
            ...     role={'new_col': StatisticRole()},
            ...     index=['a', 'b', 'c']
            ... )
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
        """Checks compatibility with another dataset.

        Internal method to verify that two datasets can be combined.

        Args:
            other (Any): Dataset to check compatibility with.

        Raises:
            ConcatDataError: If other is not a Dataset instance.
            ConcatBackendError: If backends of datasets don't match.

        Examples:
            Internal usage:
            >>> ds1 = Dataset(...)
            >>> ds2 = Dataset(...)
            >>> ds1._check_other_dataset(ds2)  # Validates compatibility
        """
        if not isinstance(other, Dataset):
            raise ConcatDataError(type(other))
        if type(other._backend) is not type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))

    def astype(
        self, dtype: Dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> "Dataset":
        """Changes the data type of specified columns.

        Args:
            dtype (Dict[str, type]): Dictionary mapping column names to target types.
            errors (Literal["raise", "ignore"]): How to handle errors:
                - "raise": Raise error if column doesn't exist or type conversion fails
                - "ignore": Skip invalid columns/conversions
                Defaults to "raise".

        Returns:
            Dataset: A new Dataset with updated data types.

        Raises:
            KeyError: If errors="raise" and a column doesn't exist.

        Examples:
            Convert column types:
            >>> ds = Dataset(...)
            >>> new_ds = ds.astype({
            ...     'col1': int,
            ...     'col2': float
            ... })

            Ignore errors:
            >>> new_ds = ds.astype({
            ...     'col1': int,
            ...     'missing_col': float
            ... }, errors='ignore')
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
        """Appends rows or columns from another dataset.

        Args:
            other (Union[Dataset, List[Dataset]]): Dataset(s) to append.
            reset_index (bool): Whether to reset index after append. Defaults to False.
            axis (int): Axis along which to append:
                - 0: Append rows (vertical concatenation)
                - 1: Append columns (horizontal concatenation)
                Defaults to 0.

        Returns:
            Dataset: New Dataset with appended data.

        Examples:
            Append rows:
            >>> ds1 = Dataset(...)
            >>> ds2 = Dataset(...)
            >>> combined = ds1.append(ds2)

            Append multiple datasets with reset index:
            >>> combined = ds1.append([ds2, ds3], reset_index=True)

            Append columns:
            >>> combined = ds1.append(ds2, axis=1)
        """
        other = Adapter.to_list(other)

        new_roles = deepcopy(self.roles)
        for o in other:
            self._check_other_dataset(o)
            new_roles.update(o.roles)

        return Dataset(
            roles=new_roles, data=self.backend.append(other, reset_index, axis)
        )

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
        """Creates a Dataset from a dictionary.

        Args:
            data (FromDictTypes): Dictionary containing data. Can be:
                - Dict[str, List]: Column name to list of values
                - Dict[str, np.ndarray]: Column name to numpy array
            roles (Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]]):
                Role definitions for columns. Can be either:
                - Dict mapping roles to column name(s)
                - Dict mapping column names to roles
            backend (BackendsEnum): Backend storage type. Defaults to pandas.
            index (Optional[Union[List, pd.Index]]): Index for the dataset.

        Returns:
            Dataset: New Dataset created from dictionary.

        Examples:
            Create from dict with roles by column:
            >>> from hypex.dataset.roles import InfoRole, StatisticRole
            >>> data = {
            ...     'id': [1, 2, 3],
            ...     'value': [10.0, 20.0, 30.0]
            ... }
            >>> roles = {
            ...     'id': InfoRole(),
            ...     'value': StatisticRole()
            ... }
            >>> ds = Dataset.from_dict(data, roles)

            Create with custom index:
            >>> ds = Dataset.from_dict(
            ...     data,
            ...     roles,
            ...     index=['a', 'b', 'c']
            ... )
        """
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend.from_dict(data, index)
        ds.data = ds._backend.data
        return ds

    def apply(
        self,
        func: Callable,
        role: Dict[str, ABCRole],
        axis: int = 0,
        **kwargs,
    ) -> "Dataset":
        """Applies a function to the dataset.

        Args:
            func (Callable): Function to apply to the data.
            role (Dict[str, ABCRole]): Role definition for resulting columns.
            axis (int): Axis to apply function along:
                - 0: Apply to each column
                - 1: Apply to each row
                Defaults to 0.
            **kwargs: Additional arguments passed to func.

        Returns:
            Dataset: New Dataset with function applied.

        Examples:
            Apply function to columns:
            >>> from hypex.dataset.roles import StatisticRole
            >>> ds = Dataset(...)
            >>> result = ds.apply(
            ...     lambda x: x * 2,
            ...     role={'doubled': StatisticRole()}
            ... )

            Apply to rows:
            >>> result = ds.apply(
            ...     lambda x: x.mean(),
            ...     role={'row_mean': StatisticRole()},
            ...     axis=1
            ... )
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
        """Applies an element-wise mapping function.

        Args:
            func (Union[Dict, Callable]): Function to apply. Can be:
                - A dictionary mapping values
                - A function that takes a single value
            na_action (Optional[str]): How to handle NA values:
                - None: Propagate NA values
                - 'ignore': Leave NA values unchanged
            **kwargs: Additional arguments passed to func.

        Returns:
            Dataset: New Dataset with mapping applied.

        Examples:
            Map using dictionary:
            >>> ds = Dataset(...)
            >>> mapping = {1: 'A', 2: 'B', 3: 'C'}
            >>> result = ds.map(mapping)

            Map using function:
            >>> result = ds.map(lambda x: x * 2)

            Handle NA values:
            >>> result = ds.map(mapping, na_action='ignore')
        """
        return Dataset(
            roles=self.roles,
            data=self._backend.map(func=func, na_action=na_action, **kwargs),
        )

    def is_empty(self) -> bool:
        """Checks if dataset is empty.

        Returns:
            bool: True if dataset has no data, False otherwise.

        Examples:
            >>> ds = Dataset(...)
            >>> if ds.is_empty():
            ...     print("Dataset is empty")
        """
        return self._backend.is_empty()

    def unique(self) -> Dict[str, List[Any]]:
        """Gets unique values for each column.

        Returns:
            Dict[str, List[Any]]: Dictionary mapping column names to lists of unique values.

        Examples:
            >>> ds = Dataset(...)
            >>> unique_values = ds.unique()
            >>> print(unique_values['column_name'])
        """
        return self._backend.unique()

    def nunique(self, dropna: bool = False) -> Dict[str, int]:
        """Counts unique values for each column.

        Args:
            dropna (bool): Whether to exclude NA values from counts.
                Defaults to False.

        Returns:
            Dict[str, int]: Dictionary mapping column names to counts of unique values.

        Examples:
            Count including NA:
            >>> ds = Dataset(...)
            >>> counts = ds.nunique()

            Count excluding NA:
            >>> counts = ds.nunique(dropna=True)
        """
        return self._backend.nunique(dropna)

    def isin(self, values: Iterable) -> "Dataset":
        """Checks whether values are contained in the dataset.

        Args:
            values (Iterable): Values to check for.

        Returns:
            Dataset: Boolean mask as Dataset where True indicates presence of value.

        Examples:
            >>> ds = Dataset(...)
            >>> mask = ds.isin([1, 2, 3])
            >>> filtered = ds[mask]
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
        """Groups dataset by values.

        Args:
            by (Any): Column(s) to group by. Can be:
                - String column name
                - List of column names
                - Dataset with single column
            func (Optional[Union[str, List]]): Aggregation function(s) to apply.
            fields_list (Optional[Union[str, List]]): Columns to include in result.
            **kwargs: Additional arguments for groupby operation.

        Returns:
            List[Tuple[str, Dataset]]: List of tuples containing:
                - Group key/name
                - Dataset containing group data

        Examples:
            Simple groupby:
            >>> ds = Dataset(...)
            >>> groups = ds.groupby('category')

            Group with aggregation:
            >>> groups = ds.groupby('category', func='mean')

            Group by multiple columns:
            >>> groups = ds.groupby(['category', 'subcategory'])

            Select specific fields:
            >>> groups = ds.groupby(
            ...     'category',
            ...     fields_list=['value1', 'value2']
            ... )
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
        """Sorts dataset by values.

        Args:
            by (Optional[MultiFieldKeyTypes]): Column(s) to sort by.
                If None, sorts by index.
            ascending (bool): Sort order. Defaults to True.
            **kwargs: Additional arguments for sort operation.

        Returns:
            Dataset: New sorted Dataset.

        Examples:
            Sort by single column:
            >>> ds = Dataset(...)
            >>> sorted_ds = ds.sort('column_name')

            Sort by multiple columns:
            >>> sorted_ds = ds.sort(['col1', 'col2'])

            Sort descending:
            >>> sorted_ds = ds.sort('column_name', ascending=False)

            Sort by index:
            >>> sorted_ds = ds.sort()
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
        """Fills NA values in the dataset.

        Args:
            values (Optional[Union[ScalarType, Dict[str, ScalarType]]]): Values to fill with:
                - Single value for all columns
                - Dict mapping column names to fill values
            method (Optional[Literal["bfill", "ffill"]]): Method for filling:
                - "bfill": Backward fill
                - "ffill": Forward fill
            **kwargs: Additional arguments for fill operation.

        Returns:
            Dataset: New Dataset with NA values filled.

        Raises:
            ValueError: If neither values nor method is provided.

        Examples:
            Fill with single value:
            >>> ds = Dataset(...)
            >>> filled = ds.fillna(0)

            Fill different values by column:
            >>> filled = ds.fillna({'col1': 0, 'col2': 'missing'})

            Forward fill:
            >>> filled = ds.fillna(method='ffill')
        """
        if values is None and method is None:
            raise ValueError("Value or filling method must be provided")
        return Dataset(
            roles=self.roles,
            data=self.backend.fillna(values=values, method=method, **kwargs),
        )

    def mean(self):
        """Calculates mean of numeric columns.

        Returns:
            Union[Dataset, float]: Mean values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> means = ds.mean()
            >>> print(means['column_name'])
        """
        return self._convert_data_after_agg(self._backend.mean())

    def max(self):
        """Calculates maximum of columns.

        Returns:
            Union[Dataset, float]: Maximum values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> maxes = ds.max()
            >>> print(maxes['column_name'])
        """
        return self._convert_data_after_agg(self._backend.max())

    def reindex(self, labels, fill_value: Optional[Any] = None) -> "Dataset":
        """Reindexes the dataset.

        Args:
            labels (Union[List, pd.Index]): New index labels.
            fill_value (Optional[Any]): Value to use for missing values.
                Defaults to None (NaN/NA).

        Returns:
            Dataset: New Dataset with updated index.

        Examples:
            >>> ds = Dataset(...)
            >>> reindexed = ds.reindex(['a', 'b', 'c'])

            With fill value:
            >>> reindexed = ds.reindex(
            ...     ['a', 'b', 'c'],
            ...     fill_value=0
            ... )
        """
        return Dataset(
            self.roles, data=self.backend.reindex(labels, fill_value=fill_value)
        )

    def idxmax(self):
        """Gets index of maximum values.

        Returns:
            Dataset: Dataset containing index labels of maximum values.

        Examples:
            >>> ds = Dataset(...)
            >>> max_indices = ds.idxmax()
            >>> print(max_indices['column_name'])
        """
        return self._convert_data_after_agg(self._backend.idxmax())

    def min(self):
        """Calculates minimum of columns.

        Returns:
            Union[Dataset, float]: Minimum values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> mins = ds.min()
            >>> print(mins['column_name'])
        """
        return self._convert_data_after_agg(self._backend.min())

    def count(self):
        """Counts non-NA values.

        Returns:
            Dataset: Count of non-NA values for each column.

        Examples:
            >>> ds = Dataset(...)
            >>> counts = ds.count()
            >>> print(counts['column_name'])
        """
        if self.is_empty():
            return Dataset.create_empty({role: InfoRole() for role in self.roles})
        return self._convert_data_after_agg(self._backend.count())

    def sum(self):
        """Calculates sum of columns.

        Returns:
            Union[Dataset, float]: Sum values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> sums = ds.sum()
            >>> print(sums['column_name'])
        """
        return self._convert_data_after_agg(self._backend.sum())

    def log(self):
        """Calculates natural logarithm.

        Returns:
            Dataset: Natural logarithm of values.

        Examples:
            >>> ds = Dataset(...)
            >>> logged = ds.log()
            >>> print(logged['column_name'])
        """
        return self._convert_data_after_agg(self._backend.log())

    def mode(self, numeric_only: bool = False, dropna: bool = True):
        """Calculate mode of columns in the dataset.

        Args:
            numeric_only (bool, optional): Whether to include only numeric columns. Defaults to False.
            dropna (bool, optional): Whether to exclude NA values. Defaults to True.

        Returns:
            Dataset: Mode values for each column with InfoRole applied.

        Examples:
            >>> ds = Dataset(...)
            >>> modes = ds.mode()
            >>> print(modes['column_name'])

            Only numeric columns:
            >>> numeric_modes = ds.mode(numeric_only=True)

            Include NA values:
            >>> modes_with_na = ds.mode(dropna=False)
        """
        t_data = self._backend.mode(numeric_only=numeric_only, dropna=dropna)
        return Dataset(data=t_data, roles={role: InfoRole() for role in t_data.columns})

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False):
        """Calculate variance of columns in the dataset.

        Args:
            skipna (bool, optional): Whether to exclude NA values. Defaults to True.
            ddof (int, optional): Delta degrees of freedom for variance calculation. Defaults to 1.
            numeric_only (bool, optional): Whether to include only numeric columns. Defaults to False.

        Returns:
            Union[Dataset, float]: Variance values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> variances = ds.var()
            >>> print(variances['column_name'])

            With custom ddof:
            >>> var_ddof0 = ds.var(ddof=0)

            Only numeric columns:
            >>> numeric_var = ds.var(numeric_only=True)
        """
        return self._convert_data_after_agg(
            self._backend.var(skipna=skipna, ddof=ddof, numeric_only=numeric_only)
        )

    def agg(self, func: Union[str, List]):
        """Aggregate using one or more operations.

        Args:
            func (Union[str, List]): Single function name as string or list of function names to apply.
                Common functions: 'sum', 'mean', 'min', 'max', 'count'

        Returns:
            Union[Dataset, float]: Aggregated results:
                - Dataset if multiple columns or functions
                - float if single column and function

        Examples:
            >>> ds = Dataset(...)
            >>> # Single aggregation
            >>> means = ds.agg('mean')
            >>> 
            >>> # Multiple aggregations
            >>> stats = ds.agg(['mean', 'std', 'max'])
            >>> print(stats['column_name'])
        """
        return self._convert_data_after_agg(self._backend.agg(func))

    def std(self, skipna: bool = True, ddof: int = 1):
        """Calculate standard deviation of columns.

        Args:
            skipna (bool, optional): Whether to exclude NA values. Defaults to True.
            ddof (int, optional): Delta degrees of freedom. Defaults to 1.

        Returns:
            Union[Dataset, float]: Standard deviation values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> stds = ds.std()
            >>> print(stds['column_name'])

            With custom ddof:
            >>> std_ddof0 = ds.std(ddof=0)
        """
        return self._convert_data_after_agg(self._backend.std(skipna=skipna, ddof=ddof))

    def quantile(self, q: float = 0.5):
        """Calculate quantiles for each column.

        Args:
            q (float, optional): Quantile to compute, between 0 and 1. Defaults to 0.5 (median).

        Returns:
            Union[Dataset, float]: Quantile values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> # Get median (50th percentile)
            >>> median = ds.quantile()
            >>> 
            >>> # Get 75th percentile
            >>> q75 = ds.quantile(0.75)
            >>> print(q75['column_name'])
        """
        return self._convert_data_after_agg(self._backend.quantile(q=q))

    def coefficient_of_variation(self):
        """Calculate coefficient of variation (CV) for each column.

        The coefficient of variation is the ratio of the standard deviation to the mean,
        expressed as a percentage.

        Returns:
            Union[Dataset, float]: CV values:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> cv = ds.coefficient_of_variation()
            >>> print(cv['column_name'])
        """
        return self._convert_data_after_agg(self._backend.coefficient_of_variation())

    def corr(self, method="pearson", numeric_only=False):
        """Calculate correlation between columns.

        Args:
            method (str, optional): Correlation method to use. Options:
                - 'pearson': Standard correlation coefficient
                - 'kendall': Kendall Tau correlation coefficient
                - 'spearman': Spearman rank correlation
                Defaults to "pearson".
            numeric_only (bool, optional): Whether to include only numeric columns. Defaults to False.

        Returns:
            Dataset: Correlation matrix with original roles preserved.

        Examples:
            >>> ds = Dataset(...)
            >>> # Pearson correlation
            >>> corr_matrix = ds.corr()
            >>> 
            >>> # Spearman correlation
            >>> spearman_corr = ds.corr(method='spearman')
            >>> print(spearman_corr)
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
        """Count unique values in each column.

        Args:
            normalize (bool, optional): Return proportions instead of counts. Defaults to False.
            sort (bool, optional): Sort by counts. Defaults to True.
            ascending (bool, optional): Sort in ascending order. Defaults to False.
            dropna (bool, optional): Whether to exclude NA values. Defaults to True.

        Returns:
            Dataset: Value counts with StatisticRole applied to count/proportion column.
                Index contains unique values, column contains counts or proportions.

        Examples:
            >>> ds = Dataset(...)
            >>> # Get raw counts
            >>> counts = ds.value_counts()
            >>> 
            >>> # Get proportions
            >>> props = ds.value_counts(normalize=True)
            >>> 
            >>> # Sort ascending and include NA
            >>> counts_asc = ds.value_counts(
            ...     ascending=True,
            ...     dropna=False
            ... )
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
        """Count NA values in each column.

        Returns:
            Union[Dataset, float]: NA counts:
                - Dataset if multiple columns
                - float if single column

        Examples:
            >>> ds = Dataset(...)
            >>> na_counts = ds.na_counts()
            >>> print(na_counts['column_name'])
        """
        return self._convert_data_after_agg(self._backend.na_counts())

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: Union[str, Iterable[str], None] = None,
        axis: Union[Literal["index", "rows", "columns"], int] = 0,
    ):
        """Remove missing values from the dataset.

        Args:
            how (Literal["any", "all"], optional): How to drop rows/columns with missing values.
                - 'any': Drop if any NA values are present
                - 'all': Drop only if all values are NA
                Defaults to "any".
            subset (Union[str, Iterable[str], None], optional): Labels of columns to check for missing values.
                None uses all columns. Defaults to None.
            axis (Union[Literal["index", "rows", "columns"], int], optional): Which axis to drop values from.
                - 0/'index'/'rows': Drop rows
                - 1/'columns': Drop columns
                Defaults to 0.

        Returns:
            Dataset: A new dataset with NA values removed according to specified criteria.

        Examples:
            >>> ds = Dataset(...)
            >>> # Drop rows with any NA
            >>> clean_ds = ds.dropna()
            >>> 
            >>> # Drop rows where all values are NA
            >>> clean_ds = ds.dropna(how='all')
            >>> 
            >>> # Drop NA values only considering certain columns
            >>> clean_ds = ds.dropna(subset=['col1', 'col2'])
            >>> 
            >>> # Drop columns with any NA values
            >>> clean_ds = ds.dropna(axis='columns')
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
        """Check for missing values in the dataset.

        Returns:
            Dataset: A boolean dataset indicating missing values (True where value is NA).

        Examples:
            >>> ds = Dataset(...)
            >>> na_mask = ds.isna()
            >>> print(na_mask['column_name'])
        """
        return self._convert_data_after_agg(self._backend.isna())

    def select_dtypes(self, include: Any = None, exclude: Any = None):
        """Select columns based on their dtype.

        Args:
            include (Any, optional): Dtypes to include. Can be string name ('float64'),
                np.dtype, or list of these. Defaults to None.
            exclude (Any, optional): Dtypes to exclude. Same format as include.
                Defaults to None.

        Returns:
            Dataset: Dataset containing only columns with selected dtypes.

        Examples:
            >>> ds = Dataset(...)
            >>> # Select numeric columns
            >>> numeric_ds = ds.select_dtypes(include=['int64', 'float64'])
            >>> 
            >>> # Exclude object columns
            >>> non_object_ds = ds.select_dtypes(exclude=['object'])
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
        """Merge this dataset with another dataset.

        Args:
            right (Dataset): Right dataset to merge with.
            on (Optional[str], optional): Column name(s) to join on if same in both datasets.
                Defaults to None.
            left_on (Optional[str], optional): Column(s) from left dataset to join on.
                Defaults to None.
            right_on (Optional[str], optional): Column(s) from right dataset to join on.
                Defaults to None.
            left_index (bool, optional): Use left index as join key. Defaults to False.
            right_index (bool, optional): Use right index as join key. Defaults to False.
            suffixes (Tuple[str, str], optional): Suffixes to add to overlapping column names.
                Defaults to ("_x", "_y").
            how (Literal["left", "right", "outer", "inner", "cross"], optional): Type of merge.
                - 'left': Keep all left rows
                - 'right': Keep all right rows
                - 'outer': Keep all rows
                - 'inner': Keep only matching rows
                - 'cross': Cartesian product
                Defaults to "inner".

        Returns:
            Dataset: Merged dataset.

        Raises:
            DataTypeError: If right is not a Dataset.
            BackendTypeError: If backends don't match.

        Examples:
            >>> ds1 = Dataset(...)
            >>> ds2 = Dataset(...)
            >>> 
            >>> # Merge on common column
            >>> merged = ds1.merge(ds2, on='id')
            >>> 
            >>> # Merge on different columns
            >>> merged = ds1.merge(
            ...     ds2,
            ...     left_on='id1',
            ...     right_on='id2'
            ... )
            >>> 
            >>> # Outer join using indices
            >>> merged = ds1.merge(
            ...     ds2,
            ...     left_index=True,
            ...     right_index=True,
            ...     how='outer'
            ... )
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
        """Drop specified labels from rows or columns.

        Args:
            labels (Any, optional): Labels to drop. Can be single label, list of labels,
                or Dataset (will use its index). Defaults to None.
            axis (int, optional): Axis to drop from:
                - 0: Drop rows
                - 1: Drop columns
                Defaults to 1.

        Returns:
            Dataset: Dataset with specified labels dropped.

        Examples:
            >>> ds = Dataset(...)
            >>> # Drop columns
            >>> ds_subset = ds.drop(['col1', 'col2'])
            >>> 
            >>> # Drop rows
            >>> ds_filtered = ds.drop([0, 1, 2], axis=0)
            >>> 
            >>> # Drop rows using another dataset's index
            >>> ds_filtered = ds.drop(other_ds, axis=0)
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
        """Filter rows or columns using specified criteria.

        Args:
            items (Optional[List], optional): List of items to include. Defaults to None.
            like (Optional[str], optional): Include labels containing this substring.
                Defaults to None.
            regex (Optional[str], optional): Include labels matching this regular expression.
                Defaults to None.
            axis (Optional[int], optional): Axis to filter:
                - 0: Filter rows
                - 1: Filter columns
                Defaults to None.

        Returns:
            Dataset: Filtered dataset.

        Examples:
            >>> ds = Dataset(...)
            >>> # Filter by list of items
            >>> filtered = ds.filter(items=['col1', 'col2'])
            >>> 
            >>> # Filter columns containing 'price'
            >>> price_cols = ds.filter(like='price')
            >>> 
            >>> # Filter using regex
            >>> numeric_cols = ds.filter(regex=r'_\d+$')
        """
        t_data = self._backend.filter(items=items, like=like, regex=regex, axis=axis)
        t_roles = {c: self.roles[c] for c in t_data.columns if c in self.roles.keys()}
        return Dataset(roles=t_roles, data=t_data)

    def dot(self, other: "Dataset") -> "Dataset":
        """Compute matrix multiplication with another dataset.

        Args:
            other (Dataset): Dataset to multiply with.

        Returns:
            Dataset: Result of matrix multiplication with roles from other dataset.

        Examples:
            >>> ds1 = Dataset(...)  # Shape (n, m)
            >>> ds2 = Dataset(...)  # Shape (m, p)
            >>> result = ds1.dot(ds2)  # Shape (n, p)
        """
        return Dataset(roles=other.roles, data=self.backend.dot(other.backend))

    def transpose(
        self,
        roles: Optional[Union[Dict[str, ABCRole], List[str]]] = None,
    ) -> "Dataset":
        """Transpose the dataset.

        Args:
            roles (Optional[Union[Dict[str, ABCRole], List[str]]], optional): New roles for transposed columns.
                Can be dict mapping column names to roles or list of column names.
                Defaults to None.

        Returns:
            Dataset: Transposed dataset.

        Examples:
            >>> ds = Dataset(...)
            >>> # Simple transpose
            >>> transposed = ds.transpose()
            >>> 
            >>> # Transpose with new roles
            >>> new_roles = {
            ...     'new_col1': InfoRole(),
            ...     'new_col2': StatisticRole()
            ... }
            >>> transposed = ds.transpose(roles=new_roles)
            >>> 
            >>> # Transpose with column names only
            >>> transposed = ds.transpose(
            ...     roles=['new_col1', 'new_col2']
            ... )
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
        """Return a random sample of the dataset.

        Args:
            frac (Optional[float], optional): Fraction of rows to sample. Defaults to None.
            n (Optional[int], optional): Number of rows to sample. Defaults to None.
            random_state (Optional[int], optional): Random seed for reproducibility.
                Defaults to None.

        Returns:
            Dataset: Random sample of rows.

        Examples:
            >>> ds = Dataset(...)
            >>> # Sample 20% of rows
            >>> sample = ds.sample(frac=0.2)
            >>> 
            >>> # Sample 100 rows
            >>> sample = ds.sample(n=100)
            >>> 
            >>> # Sample with fixed random seed
            >>> sample = ds.sample(
            ...     frac=0.1,
            ...     random_state=42
            ... )
        """
        return Dataset(
            self.roles,
            data=self.backend.sample(frac=frac, n=n, random_state=random_state),
        )

    def cov(self):
        """Calculate covariance matrix.

        Returns:
            Dataset: Covariance matrix with DefaultRole applied to all columns.

        Examples:
            >>> ds = Dataset(...)
            >>> cov_matrix = ds.cov()
            >>> print(cov_matrix)
        """
        t_data = self.backend.cov()
        return Dataset(
            {column: DefaultRole() for column in t_data.columns}, data=t_data
        )

    def rename(self, names: Dict[str, str]):
        """Rename columns.

        Args:
            names (Dict[str, str]): Dictionary mapping old names to new names.

        Returns:
            Dataset: Dataset with renamed columns, preserving roles.

        Examples:
            >>> ds = Dataset(...)
            >>> renamed = ds.rename({
            ...     'old_name1': 'new_name1',
            ...     'old_name2': 'new_name2'
            ... })
        """
        roles = {names.get(column, column): role for column, role in self.roles.items()}
        return Dataset(roles, data=self.backend.rename(names))

    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        regex: bool = False,
    ) -> "Dataset":
        """Replace values in the dataset.

        Args:
            to_replace (Any, optional): Values to replace. Can be scalar, list, dict, Series, or regex.
                Defaults to None.
            value (Any, optional): Value to replace with. Ignored if to_replace is a dict.
                Defaults to None.
            regex (bool, optional): Whether to interpret to_replace as regular expressions.
                Defaults to False.

        Returns:
            Dataset: Dataset with replaced values.

        Examples:
            >>> ds = Dataset(...)
            >>> # Replace single value
            >>> replaced = ds.replace(0, -1)
            >>> 
            >>> # Replace multiple values
            >>> replaced = ds.replace([0, 1, 2], -1)
            >>> 
            >>> # Replace using dict
            >>> replaced = ds.replace({
            ...     'old_value1': 'new_value1',
            ...     'old_value2': 'new_value2'
            ... })
            >>> 
            >>> # Replace using regex
            >>> replaced = ds.replace(
            ...     r'^old_', 
            ...     'new_',
            ...     regex=True
            ... )
        """
        return Dataset(
            self.roles,
            data=self._backend.replace(to_replace=to_replace, value=value, regex=regex),
        )

class ExperimentData:
    """A class for managing experimental data, analysis results, and metadata.

    This class provides a structured way to store and manage experimental data including
    the base dataset, additional fields, variables, groups, and analysis tables.

    Attributes:
        _data (Dataset): The base dataset for the experiment
        additional_fields (Dataset): Additional computed fields/features
        variables (Dict[str, Dict[str, Union[int, float]]]): Experiment variables/parameters
        groups (Dict[str, Dict[str, Dataset]]): Grouped data for analysis
        analysis_tables (Dict[str, Dataset]): Results of analyses
        id_name_mapping (Dict[str, str]): Mapping between IDs and names

    Examples:
        Create ExperimentData with a dataset:
        >>> ds = Dataset(...)  # Your base dataset
        >>> exp_data = ExperimentData(ds)

        Add additional fields:
        >>> exp_data.set_value(
        ...     space=ExperimentDataEnum.additional_fields,
        ...     executor_id="feature_1",
        ...     value=computed_feature
        ... )

        Store analysis results:
        >>> exp_data.set_value(
        ...     space=ExperimentDataEnum.analysis_tables,
        ...     executor_id="t_test_results",
        ...     value=t_test_df
        ... )

        Access base dataset:
        >>> base_data = exp_data.ds
    """

    def __init__(self, data: Dataset):
        """Initialize ExperimentData instance.

        Args:
            data (Dataset): Base dataset for the experiment. This will be the primary
                data source for analyses.

        Examples:
            >>> base_ds = Dataset(...)
            >>> exp_data = ExperimentData(base_ds)
        """
        self._data = data
        self.additional_fields = Dataset.create_empty(index=data.index)
        self.variables: Dict[str, Dict[str, Union[int, float]]] = {}
        self.groups: Dict[str, Dict[str, Dataset]] = {}
        self.analysis_tables: Dict[str, Dataset] = {}
        self.id_name_mapping: Dict[str, str] = {}

    @property
    def ds(self):
        """Get the base dataset.

        Returns:
            Dataset: The base dataset used for the experiment.

        Examples:
            >>> exp_data = ExperimentData(base_dataset)
            >>> base_data = exp_data.ds
            >>> print(base_data.shape)
        """
        return self._data

    @staticmethod
    def create_empty(
        roles=None, backend=BackendsEnum.pandas, index=None
    ) -> "ExperimentData":
        """Create an empty ExperimentData instance.

        Args:
            roles (Optional[Dict[str, ABCRole]]): Roles for columns in the empty dataset.
                Defaults to None.
            backend (BackendsEnum): Backend storage type to use. Defaults to pandas.
            index (Optional[List]): Index for the empty dataset. Defaults to None.

        Returns:
            ExperimentData: An empty experiment data instance.

        Examples:
            Create empty experiment data:
            >>> empty_exp = ExperimentData.create_empty()

            Create with specific roles and index:
            >>> from hypex.dataset.roles import InfoRole
            >>> roles = {'id': InfoRole()}
            >>> exp = ExperimentData.create_empty(
            ...     roles=roles,
            ...     index=['a', 'b', 'c']
            ... )
        """
        ds = Dataset.create_empty(backend, roles, index)
        return ExperimentData(ds)

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
        """Check if an executor ID exists in the specified space.

        Args:
            executor_id (int): ID to check for existence.
            space (ExperimentDataEnum): Space to check in (additional_fields,
                variables, analysis_tables, etc.).

        Returns:
            bool: True if ID exists in specified space, False otherwise.

        Examples:
            >>> exp_data = ExperimentData(base_dataset)
            >>> exists = exp_data.check_hash(
            ...     executor_id=123,
            ...     space=ExperimentDataEnum.variables
            ... )
            >>> print(exists)
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
        """Set a value in the specified experimental data space.

        Args:
            space (ExperimentDataEnum): Space to store the value in (additional_fields,
                variables, analysis_tables, groups).
            executor_id (Union[str, Dict[str, str]]): ID for the value or mapping of IDs.
            value (Any): Value to store.
            key (Optional[str]): Optional key for storing the value. Required for
                variables and groups. Defaults to None.
            role (Optional[ABCRole]): Role to assign to the value if storing in
                additional_fields. Defaults to None.

        Returns:
            ExperimentData: Self for method chaining.

        Examples:
            Add additional field:
            >>> exp_data.set_value(
            ...     space=ExperimentDataEnum.additional_fields,
            ...     executor_id="feature_1",
            ...     value=feature_values,
            ...     role=StatisticRole()
            ... )

            Store variable:
            >>> exp_data.set_value(
            ...     space=ExperimentDataEnum.variables,
            ...     executor_id="test_params",
            ...     key="alpha",
            ...     value=0.05
            ... )

            Store analysis result:
            >>> exp_data.set_value(
            ...     space=ExperimentDataEnum.analysis_tables,
            ...     executor_id="correlation_matrix",
            ...     value=corr_df
            ... )
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
        """Get IDs matching specified criteria across experimental data spaces.

        Args:
            classes (Union[type, Iterable[type], str, Iterable[str]]): Classes or class
                names to search for.
            searched_space (Union[ExperimentDataEnum, Iterable[ExperimentDataEnum], None]):
                Spaces to search in. If None, searches all spaces. Defaults to None.
            key (Optional[str]): Optional key to match against. Defaults to None.

        Returns:
            Dict[str, Dict[str, List[str]]]: Nested dictionary mapping class names to
                spaces and their matching IDs.

        Examples:
            Search for specific class:
            >>> ids = exp_data.get_ids(
            ...     classes="TestAnalysis",
            ...     searched_space=ExperimentDataEnum.analysis_tables
            ... )

            Search multiple classes across spaces:
            >>> ids = exp_data.get_ids(
            ...     classes=["FeatureGen", "Analyzer"],
            ...     searched_space=[
            ...         ExperimentDataEnum.additional_fields,
            ...         ExperimentDataEnum.analysis_tables
            ...     ]
            ... )

            Search with specific key:
            >>> ids = exp_data.get_ids(
            ...     classes="GroupAnalysis",
            ...     key="control_group"
            ... )
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
        """Get a single ID matching specified criteria.

        Args:
            class_ (Union[type, str]): Class or class name to search for.
            space (ExperimentDataEnum): Space to search in.
            key (Optional[str]): Optional key to match against. Defaults to None.

        Returns:
            str: First matching ID found.

        Raises:
            NotFoundInExperimentDataError: If no matching ID is found.

        Examples:
            Get analysis table ID:
            >>> id = exp_data.get_one_id(
            ...     class_="TTest",
            ...     space=ExperimentDataEnum.analysis_tables
            ... )

            Get ID with specific key:
            >>> id = exp_data.get_one_id(
            ...     class_="GroupAnalysis",
            ...     space=ExperimentDataEnum.groups,
            ...     key="treatment_group"
            ... )
        """
        class_ = class_ if isinstance(class_, str) else class_.__name__
        result = self.get_ids(class_, space, key)
        if (class_ not in result) or (not len(result[class_][space.value])):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

    def copy(self, data: Optional[Dataset] = None) -> "ExperimentData":
        """Create a deep copy of the ExperimentData instance.

        Args:
            data (Optional[Dataset]): Optional new base dataset to use in the copy.
                If provided, replaces the base dataset in the copy. Defaults to None.

        Returns:
            ExperimentData: A deep copy of the current instance.

        Examples:
            Simple copy:
            >>> exp_data_copy = exp_data.copy()

            Copy with new base dataset:
            >>> new_base = Dataset(...)
            >>> exp_data_copy = exp_data.copy(data=new_base)
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
        """Search for fields with specified roles across the dataset.

        Args:
            roles (Union[ABCRole, Iterable[ABCRole]]): Role(s) to search for.
            tmp_role (bool): Whether to include temporary roles in search.
                Defaults to False.
            search_types (Optional[List[type]]): Types to filter results by.
                Defaults to None.

        Returns:
            List[str]: List of field names matching the search criteria.

        Examples:
            Search for specific role:
            >>> from hypex.dataset.roles import StatisticRole
            >>> fields = exp_data.field_search(StatisticRole())

            Search multiple roles:
            >>> from hypex.dataset.roles import InfoRole, FilterRole
            >>> fields = exp_data.field_search([InfoRole(), FilterRole()])

            Search with type filtering:
            >>> fields = exp_data.field_search(
            ...     StatisticRole(),
            ...     search_types=[float, int]
            ... )
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
        """Search for and return data with specified roles.

        Args:
            roles (Union[ABCRole, Iterable[ABCRole]]): Role(s) to search for.
            tmp_role (bool): Whether to include temporary roles in search.
                Defaults to False.
            search_types (Optional[List[type]]): Types to filter results by.
                Defaults to None.

        Returns:
            Dataset: Dataset containing all matching fields.

        Examples:
            Get numeric statistics:
            >>> from hypex.dataset.roles import StatisticRole
            >>> stats_data = exp_data.field_data_search(
            ...     StatisticRole(),
            ...     search_types=[float, int]
            ... )

            Get all info fields:
            >>> from hypex.dataset.roles import InfoRole
            >>> info_data = exp_data.field_data_search(InfoRole())

            Get multiple role types:
            >>> from hypex.dataset.roles import FilterRole, AdditionalRole
            >>> mixed_data = exp_data.field_data_search([
            ...     FilterRole(),
            ...     AdditionalRole()
            ... ])
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
    """Adapter class for converting various data types to Dataset objects.

    This class provides static methods to convert different data structures
    (dictionaries, DataFrames, lists, scalar values) into Dataset objects
    while preserving role information.

    Examples:
        Convert dictionary to Dataset:
        >>> data = {'value': [1, 2, 3]}
        >>> roles = {'value': StatisticRole()}
        >>> ds = DatasetAdapter.to_dataset(data, roles)

        Convert scalar to Dataset:
        >>> value = 42
        >>> ds = DatasetAdapter.to_dataset(value, StatisticRole())

        Convert DataFrame to Dataset:
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        >>> roles = {'col1': StatisticRole(), 'col2': InfoRole()}
        >>> ds = DatasetAdapter.to_dataset(df, roles)
    """

    @staticmethod
    def to_dataset(
        data: Union[Dict, Dataset, pd.DataFrame, List, str, int, float, bool],
        roles: Union[ABCRole, Dict[str, ABCRole]],
    ) -> Dataset:
        """Convert various data types to a Dataset object.

        Args:
            data (Union[Dict, Dataset, pd.DataFrame, List, str, int, float, bool]):
                Data to convert to Dataset format.
            roles (Union[ABCRole, Dict[str, ABCRole]]): Roles for the dataset columns.
                Can be single role or role mapping.

        Returns:
            Dataset: Converted dataset.

        Raises:
            InvalidArgumentError: If data type is not supported or roles format
                doesn't match data type.

        Examples:
            Convert dictionary:
            >>> data = {'value': [1, 2, 3]}
            >>> ds = DatasetAdapter.to_dataset(
            ...     data,
            ...     {'value': StatisticRole()}
            ... )

            Convert scalar with single role:
            >>> ds = DatasetAdapter.to_dataset(42, StatisticRole())

            Convert DataFrame:
            >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            >>> ds = DatasetAdapter.to_dataset(
            ...     df,
            ...     {'a': StatisticRole(), 'b': StatisticRole()}
            ... )
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
        """Convert a scalar value to a Dataset.

        Args:
            data (ScalarType): Scalar value to convert (str, int, float, bool).
            roles (Union[ABCRole, Dict[str, ABCRole]]): Role(s) for the dataset.

        Returns:
            Dataset: Dataset containing the scalar value.

        Examples:
            >>> value = 42
            >>> ds = DatasetAdapter.value_to_dataset(
            ...     value,
            ...     StatisticRole()
            ... )

            >>> text = "example"
            >>> ds = DatasetAdapter.value_to_dataset(
            ...     text,
            ...     {'text': InfoRole()}
            ... )
        """
        if isinstance(roles, ABCRole):
            roles = {"value": roles}
        return Dataset(roles=roles, data=pd.DataFrame({list(roles.keys())[0]: [data]}))

    @staticmethod
    def dict_to_dataset(
        data: Dict, roles: Union[ABCRole, Dict[str, ABCRole]]
    ) -> Dataset:
        """Convert a dictionary to a Dataset.

        Args:
            data (Dict): Dictionary to convert.
            roles (Union[ABCRole, Dict[str, ABCRole]]): Role(s) for the dataset columns.

        Returns:
            Dataset: Dataset created from dictionary.

        Examples:
            Single role for all columns:
            >>> data = {'col1': [1, 2], 'col2': [3, 4]}
            >>> ds = DatasetAdapter.dict_to_dataset(
            ...     data,
            ...     StatisticRole()
            ... )

            Different roles per column:
            >>> data = {'id': [1, 2], 'value': [10, 20]}
            >>> roles = {
            ...     'id': InfoRole(),
            ...     'value': StatisticRole()
            ... }
            >>> ds = DatasetAdapter.dict_to_dataset(data, roles)
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
        """Convert a list to a Dataset.

        Args:
            data (List): List to convert.
            roles (Dict[str, ABCRole]): Roles mapping for the dataset columns.

        Returns:
            Dataset: Dataset created from list.

        Examples:
            >>> data = [1, 2, 3, 4]
            >>> roles = {'values': StatisticRole()}
            >>> ds = DatasetAdapter.list_to_dataset(data, roles)

            >>> data = ['a', 'b', 'c']
            >>> roles = {'category': InfoRole()}
            >>> ds = DatasetAdapter.list_to_dataset(data, roles)
        """
        return Dataset(
            roles=roles,
            data=pd.DataFrame(data=data, columns=[list(roles.keys())[0]]),
        )

    @staticmethod
    def frame_to_dataset(data: pd.DataFrame, roles: Dict[str, ABCRole]) -> Dataset:
        """Convert a pandas DataFrame to a Dataset.

        Args:
            data (pd.DataFrame): DataFrame to convert.
            roles (Dict[str, ABCRole]): Roles mapping for the dataset columns.

        Returns:
            Dataset: Dataset created from DataFrame.

        Examples:
            >>> df = pd.DataFrame({
            ...     'id': [1, 2, 3],
            ...     'value': [10.0, 20.0, 30.0]
            ... })
            >>> roles = {
            ...     'id': InfoRole(),
            ...     'value': StatisticRole()
            ... }
            >>> ds = DatasetAdapter.frame_to_dataset(df, roles)
        """
        return Dataset(
            roles=roles,
            data=data,
        )
