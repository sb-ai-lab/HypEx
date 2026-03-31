from __future__ import annotations

import copy

from typing import Any, Callable, TYPE_CHECKING

from ..utils import NAME_BORDER_SYMBOL

from .roles import (
    ABCRole,
    InfoRole,
    StatisticRole
)

if TYPE_CHECKING:
    from .abstract import DatasetBase


class GroupedDataset:
    """
    Represents a dataset that has been grouped by specific columns.

    This class provides an interface to perform aggregation and transformation
    operations on groups within a dataset. It wraps a backend groupby object
    (e.g., from pandas or another engine) and ensures that column roles are
    correctly maintained or updated after operations.

    Attributes:
        _groupby (Any): The underlying backend groupby object.
        _dataset_class (type[DatasetBase]): The class used to instantiate resulting datasets.
        roles (dict[str, ABCRole]): Mapping of column names to their semantic roles.
        tmp_roles (dict[str, ABCRole]): Temporary roles mapping.
        _group_cols (list[str]): Columns used for grouping.
        _backend_data (Any): Original DataSet.
    """
    def __init__(self,
                 backend_groupby: Any, 
                 dataset_class: type[DatasetBase], 
                 roles: dict[str, ABCRole], 
                 tmp_roles: dict[str, ABCRole], 
                 group_cols: list[str] | None=None,
                 backend_data: Any = None):
        """
        Initialize the GroupedDataset.

        Args:
            backend_groupby: The underlying groupby object from the data backend.
            dataset_class: The class constructor for creating new Dataset instances.
            roles: Dictionary mapping column names to their assigned roles.
            tmp_roles: Dictionary mapping column names to temporary roles.
            group_cols: List of column names used to group the data. Defaults to empty list.
            backend_data: The raw backend data structure, required for iteration. Defaults to None.
        """
        self._groupby = backend_groupby
        self._dataset_class = dataset_class
        self.roles = roles
        self.tmp_roles = tmp_roles
        self._group_cols = group_cols if group_cols is not None else []
        self._backend_data = backend_data

    def _get_agg_roles(self, 
                       result_columns: list[str]) -> dict[str, ABCRole]:
        """
        Determine the roles for columns resulting from an aggregation operation.

        If a column exists in the original roles, its role is deep-copied.
        Otherwise, a default StatisticRole is assigned.

        Args:
            result_columns: List of column names present after aggregation.

        Returns:
            A dictionary mapping column names to their corresponding role instances.
        """
        new_roles = {}
        for col in result_columns:
            if col in self.roles:
                new_roles[col] = copy.deepcopy(self.roles[col])
            else:
                new_roles[col] = StatisticRole()
        return new_roles

    def _execute_agg(self, 
                     func: str | dict[str, str] | list[str]) -> Any:
        """
        Execute the aggregation function on the backend groupby object.

        Handles different backend types (objects with .agg method or lists of groups).

        Args:
            func: The aggregation function(s) to apply. Can be a string, 
                  a dictionary mapping columns to functions, or a list of functions.

        Returns:
            The aggregated data object from the backend.

        Raises:
            TypeError: If the groupby object type is unsupported.
        """
        if hasattr(self._groupby, 'agg'):
            return self._groupby.agg(func)
        
        elif isinstance(self._groupby, list):
            aggregated_groups = []
            for key, group_df in self._groupby:
                if hasattr(group_df, 'agg'):
                    agg_res = group_df.agg(func)
                else:
                    agg_res = group_df.agg(func)
                aggregated_groups.append(agg_res)
            
            if not aggregated_groups:
                return None
            result_data = self._dataset_class._backend.concat(aggregated_groups)
            return result_data
            
        else:
            raise TypeError(f"Unsupported groupby object type: {type(self._groupby)}")

    def agg(self,
            func: str | dict[str, str] | list[str]) -> DatasetBase:
        """
        Aggregate groups using the specified function(s).

        Performs aggregation, handles column renaming for multi-level indices,
        drops fully null columns, and assigns appropriate roles to the result.

        Args:
            func: Function, list of functions, or dictionary mapping columns to 
                  functions to apply during aggregation.

        Returns:
            A new DatasetBase instance containing the aggregated data and updated roles.
        """
        result_data = self._execute_agg(func)

        if result_data is None:
            return self._dataset_class(roles={}, data=None)

        if isinstance(func, list) and hasattr(result_data, 'columns'):
            if hasattr(result_data.columns, 'levels'):  # MultiIndex from list agg
                result_data.columns = [
                    f"{col}{NAME_BORDER_SYMBOL}{stat}"
                    for col, stat in result_data.columns
                ]

        if hasattr(result_data, 'columns'):
            try:
                if hasattr(result_data, 'isnull') and hasattr(result_data, 'drop'):
                    null_mask = result_data.isnull().all()
                    cols_to_drop = [col for col, is_null in null_mask.items() if is_null]
                    if cols_to_drop:
                        result_data = result_data.drop(columns=cols_to_drop)
            except (AttributeError, KeyError, TypeError) as e:
                raise type(e)(f"Could not drop fully null columns: {e}") from e
            
            result_columns = list(result_data.columns)
            new_roles = self._get_agg_roles(result_columns)
        else:
            new_roles = {}

        return self._dataset_class(roles=new_roles, data=result_data)

    def apply(self, 
              func: Callable[..., Any]) -> DatasetBase:
        """
        Apply a custom function to each group and combine the results.

        Applies the callable to each group independently and concatenates the results.
        Resulting columns are assigned InfoRole by default.

        Args:
            func: A callable to apply to each group.

        Returns:
            A new DatasetBase instance containing the transformed data.

        Raises:
            NotImplementedError: If apply is not supported for the current groupby type.
        """
        if hasattr(self._groupby, 'apply'):
            result_data = self._groupby.apply(func)
        elif isinstance(self._groupby, list):
            results = []
            for key, group_df in self._groupby:
                res = group_df.apply(func)
                results.append(res)
            if not results:
                return None
            result_data = self._dataset_class._backend.concat(results)
        else:
            raise NotImplementedError("Apply not supported for this groupby type")
            
        if hasattr(result_data, 'columns'):
            new_roles = {col: InfoRole() for col in result_data.columns}
        else:
            new_roles = {}
            
        return self._dataset_class(roles=new_roles, data=result_data)

    def count(self) -> DatasetBase:
        """
        Compute the count of non-NA cells for each group.

        Returns:
            A DatasetBase instance containing the count aggregation.
        """
        return self.agg("count")

    def sum(self, *cols: str) -> DatasetBase:
        """
        Compute the sum of values for each group.

        Args:
            *cols: Optional specific column names to sum. If none provided, 
                   applies to all applicable columns.

        Returns:
            A DatasetBase instance containing the sum aggregation.
        """
        if cols:
            return self.agg({col: 'sum' for col in cols})
        return self.agg("sum")

    def mean(self, *cols: str) -> DatasetBase:
        """
        Compute the mean of values for each group.

        Args:
            *cols: Optional specific column names to average.

        Returns:
            A DatasetBase instance containing the mean aggregation.
        """
        if cols:
            return self.agg({col: 'mean' for col in cols})
        return self.agg("mean")

    def min(self, *cols: str) -> DatasetBase:
        """
        Compute the minimum value for each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the min aggregation.
        """
        if cols:
            return self.agg({col: 'min' for col in cols})
        return self.agg("min")

    def max(self, *cols: str) -> DatasetBase:
        """
        Compute the maximum value for each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the max aggregation.
        """
        if cols:
            return self.agg({col: 'max' for col in cols})
        return self.agg("max")

    def first(self, *cols: str) -> DatasetBase:
        """
        Get the first value in each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the first value aggregation.
        """
        if cols:
            return self.agg({col: 'first' for col in cols})
        return self.agg("first")

    def last(self, *cols: str) -> DatasetBase:
        """
        Get the last value in each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the last value aggregation.
        """
        if cols:
            return self.agg({col: 'last' for col in cols})
        return self.agg("last")

    def std(self, *cols: str) -> DatasetBase:
        """
        Compute the standard deviation for each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the std aggregation.
        """
        if cols:
            return self.agg({col: 'std' for col in cols})
        return self.agg("std")

    def var(self, *cols: str) -> DatasetBase:
        """
        Compute the variance for each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the var aggregation.
        """
        if cols:
            return self.agg({col: 'var' for col in cols})
        return self.agg("var")

    def median(self, *cols: str) -> DatasetBase:
        """
        Compute the median value for each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the median aggregation.
        """
        if cols:
            return self.agg({col: 'median' for col in cols})
        return self.agg("median")

    def prod(self, *cols: str) -> DatasetBase:
        """
        Compute the product of values for each group.

        Args:
            *cols: Optional specific column names.

        Returns:
            A DatasetBase instance containing the product aggregation.
        """
        if cols:
            return self.agg({col: 'prod' for col in cols})
        return self.agg("prod")

    def value_counts(self, *cols: str, _add_suffix: bool = False) -> DatasetBase:
        feature_cols = list(cols) if cols else [
            c for c in self.roles if c not in self._group_cols
        ]

        raw = self._backend_data.grouped_value_counts(self._group_cols, feature_cols)

        suffix = f"{NAME_BORDER_SYMBOL}value_counts" if _add_suffix else ""
        if suffix:
            raw["data"] = {f"{col}{suffix}": v for col, v in raw["data"].items()}

        new_roles = {col: StatisticRole() for col in raw["data"]}
        return self._dataset_class(roles=new_roles, data=raw)

    def size(self) -> DatasetBase:
        """
        Compute the size of each group.

        Returns:
            A DatasetBase instance with a single 'size' column representing 
            the number of rows in each group.
        """
        result = self._groupby.size() if hasattr(self._groupby, 'size') else self.agg("count")
        if hasattr(result, 'to_frame'):
            result = result.to_frame('size')
        return self._dataset_class(roles={'size': StatisticRole()}, data=result)

    def __iter__(self):
        """
        Iterate over the groups in the dataset.

        Yields:
            tuple: A tuple containing (group_key, DatasetBase), where group_key 
                   is the grouping value(s) and DatasetBase is the subset of data 
                   for that group.

        Raises:
            TypeError: If the backend data or group columns are not set, 
                       indicating improper initialization.
        """
        if self._backend_data is None or not self._group_cols:
            raise TypeError(
                "GroupedDataset is not iterable: backend or group_cols not set. "
                "Use Dataset.groupby() instead of constructing GroupedDataset directly."
            )
        for key, group in self._backend_data.iter_groups(self._group_cols):
            yield key, self._dataset_class(roles=self.roles, data=group)