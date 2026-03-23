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
    def __init__(self,
                 backend_groupby: Any, 
                 dataset_class: type[DatasetBase], 
                 roles: dict[str, ABCRole], 
                 tmp_roles: dict[str, ABCRole], 
                 group_cols: list[str] | None=None):
        self._groupby = backend_groupby
        self._dataset_class = dataset_class
        self.roles = roles
        self.tmp_roles = tmp_roles
        self._group_cols = group_cols if group_cols is not None else []

    def _get_agg_roles(self, 
                       result_columns: list[str]) -> dict[str, ABCRole]:
        new_roles = {}
        for col in result_columns:
            if col in self.roles:
                new_roles[col] = copy.deepcopy(self.roles[col])
            else:
                new_roles[col] = StatisticRole()
        return new_roles

    def _execute_agg(self, 
                     func: str | dict[str, str] | list[str]) -> Any:
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
            result_columns = list(result_data.columns)
            new_roles = self._get_agg_roles(result_columns)
        else:
            new_roles = {}

        return self._dataset_class(roles=new_roles, data=result_data)

    def apply(self, 
              func: Callable[..., Any]) -> DatasetBase:
        if hasattr(self._groupby, 'apply'):
            result_data = self._groupby.apply(func)
        elif isinstance(self._groupby, list):
            results = []
            for key, group_df in self._groupby:
                res = group_df.apply(func)
                results.append(res)
            result_data = ps.concat(results)
        else:
            raise NotImplementedError("Apply not supported for this groupby type")
            
        if hasattr(result_data, 'columns'):
            new_roles = {col: InfoRole() for col in result_data.columns}
        else:
            new_roles = {}
            
        return self._dataset_class(roles=new_roles, data=result_data)

    def count(self) -> DatasetBase:
        return self.agg("count")

    def sum(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'sum' for col in cols})
        return self.agg("sum")

    def mean(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'mean' for col in cols})
        return self.agg("mean")

    def min(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'min' for col in cols})
        return self.agg("min")

    def max(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'max' for col in cols})
        return self.agg("max")

    def first(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'first' for col in cols})
        return self.agg("first")

    def last(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'last' for col in cols})
        return self.agg("last")

    def std(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'std' for col in cols})
        return self.agg("std")

    def var(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'var' for col in cols})
        return self.agg("var")

    def median(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'median' for col in cols})
        return self.agg("median")

    def prod(self, *cols: str) -> DatasetBase:
        if cols:
            return self.agg({col: 'prod' for col in cols})
        return self.agg("prod")

    def size(self) -> DatasetBase:
        result = self._groupby.size() if hasattr(self._groupby, 'size') else self.agg("count")
        if hasattr(result, 'to_frame'):
            result = result.to_frame('size')
        return self._dataset_class(roles={'size': StatisticRole()}, data=result)

    def __iter__(self) -> Iterable[tuple[Any, DatasetBase]]:
        if isinstance(self._groupby, list):
            for key, data in self._groupby:
                yield key, self._dataset_class(roles=self.roles, data=data)
        elif hasattr(self._groupby, '__iter__'):
            for key, group in self._groupby:
                yield key, self._dataset_class(roles=self.roles, data=group)
        else:
            raise TypeError("Grouped object is not iterable")