from __future__ import annotations

import warnings
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Callable, Hashable, Literal, Sequence

import numpy as np
import pandas as pd  # type: ignore
from numpy import ndarray

from ..utils import (
    ID_SPLIT_SYMBOL,
    BackendsEnum,
    BackendTypeError,
    ConcatBackendError,
    ConcatDataError,
    DataTypeError,
    ExperimentDataEnum,
    FromDictTypes,
    MultiFieldKeyTypes,
    NotFoundInExperimentDataError,
    ScalarType,
)
from ..utils.adapter import Adapter
from ..utils.errors import InvalidArgumentError
from .abstract import DatasetBase
from .roles import (
    ABCRole,
    AdditionalRole,
    DefaultRole,
    FilterRole,
    InfoRole,
    StatisticRole,
)


class Dataset(DatasetBase):
    class Locker:
        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item) -> Dataset:
            t_data = self.backend.loc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            column_name = item[1]
            column_data_type = self.roles[column_name].data_type
            if (
                column_data_type is None
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
        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item) -> Dataset:
            t_data = self.backend.iloc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            column_index = item[1]
            column_name = self.backend.data.columns[column_index]
            column_data_type = self.roles[column_name].data_type
            if (
                column_data_type is None
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
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole],
        data: pd.DataFrame | str | None = None,
        backend: BackendsEnum | None = None,
        default_role: ABCRole | None = None,
    ):
        super().__init__(roles, data, backend, default_role)
        self.loc = self.Locker(self._backend, self.roles)
        self.iloc = self.ILocker(self._backend, self.roles)

    def __getitem__(self, item: Iterable | str | int) -> Dataset:
        if isinstance(item, Dataset):
            item = item.data
        items = (
            [item] if isinstance(item, str) or not isinstance(item, Iterable) else item
        )
        roles: dict = {
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
                column_data_type is None
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
        return self.__binary_magic_operator(other=other, func_name="__eq__")

    def __ne__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__ne__")

    def __le__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__le__")

    def __lt__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__lt__")

    def __ge__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__ge__")

    def __gt__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__gt__")

    # unary operators:
    def __pos__(self):
        return Dataset(roles=self.roles, data=(+self._backend))

    def __neg__(self):
        return Dataset(roles=self.roles, data=(-self._backend))

    def __abs__(self):
        return Dataset(roles=self.roles, data=abs(self._backend))

    def __invert__(self):
        return Dataset(roles=self.roles, data=(~self._backend))

    def __round__(self, ndigits: int = 0):
        return Dataset(roles=self.roles, data=round(self._backend, ndigits))

    def __bool__(self):
        return not self._backend.is_empty()

    # Binary math operators:
    def __add__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__add__")

    def __sub__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__sub__")

    def __mul__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__mul__")

    def __floordiv__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__floordiv__")

    def __div__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__div__")

    def __truediv__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__truediv__")

    def __mod__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__mod__")

    def __pow__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__pow__")

    def __and__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__and__")

    def __or__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__or__")

    # Right math operators:
    def __radd__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__radd__")

    def __rsub__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__rsub__")

    def __rmul__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__rmul__")

    def __rfloordiv__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__rfloordiv__")

    def __rdiv__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__rdiv__")

    def __rtruediv__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__rtruediv__")

    def __rmod__(self, other):
        return self.__binary_magic_operator(other=other, func_name="__rmod__")

    def __rpow__(self, other) -> Any:
        return self.__binary_magic_operator(other=other, func_name="__rpow__")

    @property
    def index(self):
        return self.backend.index

    @index.setter
    def index(self, value):
        self.backend.data.index = value

    @property
    def data(self):
        return self._backend.data

    @data.setter
    def data(self, value):
        self.backend.data = value

    @property
    def columns(self):
        return self.backend.columns

    @staticmethod
    def create_empty(roles=None, index=None, backend=BackendsEnum.pandas) -> Dataset:
        if roles is None:
            roles = {}
        index = [] if index is None else index
        columns = list(roles.keys())
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend.create_empty(index, columns)
        ds.data = ds.backend.data
        return ds

    def _convert_data_after_agg(self, result) -> Dataset | float:
        if isinstance(result, float):
            return result
        role: ABCRole = StatisticRole()
        return Dataset(data=result, roles={column: role for column in result.columns})

    def get(
        self,
        key,
        default=None,
    ) -> Dataset:
        res = self._backend.get(key, default)
        return Dataset(data=self._backend.get(key, default), roles=deepcopy(self.roles))

    def take(
        self,
        indices: int | list[int],
        axis: Literal["index", "columns", "rows"] | int = 0,
    ) -> Dataset:
        new_data = self._backend.take(indices=indices, axis=axis)
        new_roles = (
            {k: deepcopy(v) for k, v in self.roles.items() if k in new_data.columns}
            if axis == 1
            else deepcopy(self.roles)
        )
        return Dataset(data=new_data, roles=new_roles)

    def add_column(
        self,
        data,
        role: dict[str, ABCRole] | None = None,
        index: Iterable[Hashable] | None = None,
    ):
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
            if isinstance(role, dict) and any(
                [not isinstance(r, ABCRole) for r in role.values()]
            ):
                raise TypeError("Role values must be of type ABCRole")
            if isinstance(data, Dataset):
                data = data.data
            self.roles.update(role)
            self._backend.add_column(data, list(role.keys()), index)
        return self

    def _check_other_dataset(self, other):
        if not isinstance(other, Dataset):
            raise ConcatDataError(type(other))
        if type(other._backend) is not type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))

    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> Dataset:
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

    def append(self, other, reset_index=False, axis=0) -> Dataset:
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
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole],
        backend: BackendsEnum = BackendsEnum.pandas,
        index=None,
    ) -> Dataset:
        ds = Dataset(roles=roles, backend=backend)
        # if all([isinstance(v, Dataset) for v in data.values()]):
        #     ds._backend = ds._backend.from_dict({k: v.data for k, v in data.items()}, data, index)
        # else:
        ds._backend = ds._backend.from_dict(data, index)
        ds.data = ds._backend.data
        return ds

    # What is going to happen when a matrix is returned?
    def apply(
        self,
        func: Callable,
        role: dict[str, ABCRole],
        axis: int = 0,
        **kwargs,
    ) -> Dataset:
        if self.is_empty():
            return deepcopy(self)
        tmp_data = self._backend.apply(
            func=func, axis=axis, column_name=next(iter(role.keys())), **kwargs
        )
        tmp_roles = (
            {next(iter(role.keys())): next(iter(role.values()))}
            if ((not tmp_data.any().any()) and len(role) > 1)
            else role
        )
        return Dataset(
            data=tmp_data,
            roles=tmp_roles,
        )

    def map(self, func, na_action=None, **kwargs) -> Dataset:
        return Dataset(
            roles=self.roles,
            data=self._backend.map(func=func, na_action=na_action, **kwargs),
        )

    def is_empty(self) -> bool:
        return self._backend.is_empty()

    def unique(self) -> dict[str, list[Any]]:
        return self._backend.unique()

    def nunique(self, dropna: bool = False) -> dict[str, int]:
        return self._backend.nunique(dropna)

    def isin(self, values: Iterable) -> Dataset:
        role: ABCRole = FilterRole()
        return Dataset(
            roles={column: role for column in self.roles.keys()},
            data=self._backend.isin(values),
        )

    def groupby(
        self,
        by: Any,
        func: str | list | None = None,
        fields_list: str | list | None = None,
        reset_index: bool = True,
        **kwargs,
    ) -> list[tuple[str, Dataset]]:
        if isinstance(by, Dataset) and len(by.columns) == 1:
            # if reset_index:
            #     self.data = self.data.reset_index(drop=True)
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
        by: MultiFieldKeyTypes | None = None,
        ascending: bool = True,
        **kwargs,
    ):
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
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        **kwargs,
    ):
        if values is None and method is None:
            raise ValueError("Value or filling method must be provided")
        return Dataset(
            roles=self.roles,
            data=self.backend.fillna(values=values, method=method, **kwargs),
        )

    def mean(self):
        return self._convert_data_after_agg(self._backend.mean())

    def max(self):
        return self._convert_data_after_agg(self._backend.max())

    def reindex(self, labels, fill_value: Any | None = None) -> Dataset:
        return Dataset(
            self.roles, data=self.backend.reindex(labels, fill_value=fill_value)
        )

    def idxmax(self):
        return self._convert_data_after_agg(self._backend.idxmax())

    def min(self):
        return self._convert_data_after_agg(self._backend.min())

    def count(self):
        if self.is_empty():
            return Dataset.create_empty({role: InfoRole() for role in self.roles})
        return self._convert_data_after_agg(self._backend.count())

    def sum(self):
        return self._convert_data_after_agg(self._backend.sum())

    def log(self):
        return self._convert_data_after_agg(self._backend.log())

    def mode(self, numeric_only: bool = False, dropna: bool = True):
        t_data = self._backend.mode(numeric_only=numeric_only, dropna=dropna)
        return Dataset(data=t_data, roles={role: InfoRole() for role in t_data.columns})

    def var(self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False):
        return self._convert_data_after_agg(
            self._backend.var(skipna=skipna, ddof=ddof, numeric_only=numeric_only)
        )

    def agg(self, func: str | list):
        return self._convert_data_after_agg(self._backend.agg(func))

    def std(self, skipna: bool = True, ddof: int = 1):
        return self._convert_data_after_agg(self._backend.std(skipna=skipna, ddof=ddof))

    def quantile(self, q: float = 0.5):
        return self._convert_data_after_agg(self._backend.quantile(q=q))

    def coefficient_of_variation(self):
        return self._convert_data_after_agg(self._backend.coefficient_of_variation())

    def corr(self, method="pearson", numeric_only=False):
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
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ):
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
        return self._convert_data_after_agg(self._backend.isna())

    def select_dtypes(self, include: Any = None, exclude: Any = None):
        # Filter data by dtypes
        t_data = self._backend.select_dtypes(include=include, exclude=exclude)

        # Keep only roles for remaining columns
        t_roles = {k: v for k, v in self.roles.items() if k in t_data.columns}
        return Dataset(roles=t_roles, data=t_data)

    def merge(
        self,
        right,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
    ):
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
        items: list | None = None,
        like: str | None = None,
        regex: str | None = None,
        axis: int | None = None,
    ) -> Dataset:
        t_data = self._backend.filter(items=items, like=like, regex=regex, axis=axis)
        t_roles = {c: self.roles[c] for c in t_data.columns if c in self.roles.keys()}
        return Dataset(roles=t_roles, data=t_data)

    def dot(self, other: Dataset | ndarray) -> Dataset:
        return Dataset(
            roles=deepcopy(other.roles) if isinstance(other, Dataset) else {},
            data=self.backend.dot(
                other.backend if isinstance(other, Dataset) else other
            ),
        )

    def transpose(
        self,
        roles: dict[str, ABCRole] | list[str] | None = None,
    ) -> Dataset:
        # Get role names if provided
        roles_names: list[str | None] = (
            list(roles.keys()) or [] if isinstance(roles, dict) else roles
        )

        # Transpose data
        result_data = self.backend.transpose(roles_names)

        # Create default roles if none provided
        if roles is None or isinstance(roles, list):
            names = result_data.columns if roles is None else roles
            roles = {column: DefaultRole() for column in names}

        return Dataset(roles=roles, data=result_data)

    def sample(
        self,
        frac: float | None = None,
        n: int | None = None,
        random_state: int | None = None,
    ) -> Dataset:
        return Dataset(
            self.roles,
            data=self.backend.sample(frac=frac, n=n, random_state=random_state),
        )

    def cov(self):
        t_data = self.backend.cov()
        return Dataset(
            {column: DefaultRole() for column in t_data.columns}, data=t_data
        )

    def rename(self, names: dict[str, str]):
        roles = {names.get(column, column): role for column, role in self.roles.items()}
        return Dataset(roles, data=self.backend.rename(names))

    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        regex: bool = False,
    ) -> Dataset:
        return Dataset(
            self.roles,
            data=self._backend.replace(to_replace=to_replace, value=value, regex=regex),
        )

    def list_to_columns(self, column: str) -> Dataset:
        if not pd.api.types.is_list_like(self.backend[column][0]):
            return self
        extended_data = self.backend.list_to_columns(column)
        extended_roles = {
            c: deepcopy(self.roles[column]) for c in extended_data.columns
        }
        extended_ds = Dataset(roles=extended_roles, data=extended_data)
        return self.append(extended_ds, axis=1).drop(column, axis=1)


class ExperimentData:
    def __init__(self, data: Dataset):
        self._data = data
        self.additional_fields = Dataset.create_empty(index=data.index)
        self.variables: dict[str, dict[str, int | float]] = {}
        self.groups: dict[str, dict[str, Dataset]] = {}
        self.analysis_tables: dict[str, Dataset] = {}
        self.id_name_mapping: dict[str, str] = {}

    @property
    def ds(self):
        """
        Get the base dataset.
        """
        return self._data

    @staticmethod
    def create_empty(
        roles=None, backend=BackendsEnum.pandas, index=None
    ) -> ExperimentData:
        ds = Dataset.create_empty(backend, roles, index)
        return ExperimentData(ds)

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
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
        executor_id: str | dict[str, str],
        value: Any,
        key: str | None = None,
        role=None,
    ) -> ExperimentData:
        # Handle additional fields
        if space == ExperimentDataEnum.additional_fields:
            if not isinstance(value, Dataset):
                self.additional_fields = self.additional_fields.add_column(
                    data=value, role={executor_id: role}
                )
            elif len(value.columns) == 1:
                role = role[0] if isinstance(role, list) else role
                role = list(role.values())[0] if isinstance(role, dict) else role
                executor_id = (
                    executor_id[0] if isinstance(executor_id, list) else executor_id
                )
                executor_id = (
                    list(executor_id.keys())[0]
                    if isinstance(executor_id, dict)
                    else executor_id
                )
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
            elif isinstance(value, dict):
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
        classes: type | Iterable[type] | str | Iterable[str],
        searched_space: ExperimentDataEnum | Iterable[ExperimentDataEnum] | None = None,
        key: str | None = None,
    ) -> dict[str, dict[str, list[str]]]:
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
        class_: type | str,
        space: ExperimentDataEnum,
        key: str | None = None,
    ) -> str:
        class_ = class_ if isinstance(class_, str) else class_.__name__
        result = self.get_ids(class_, space, key)
        if (class_ not in result) or (not len(result[class_][space.value])):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

    def copy(self, data: Dataset | None = None) -> ExperimentData:
        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result

    def field_search(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types=None,
    ) -> list[str]:
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
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types=None,
    ) -> Dataset:
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
        if not searched_data.is_empty():
            searched_data.index = self.ds.index
        return searched_data


class DatasetAdapter(Adapter):
    @staticmethod
    def to_dataset(
        data: dict | Dataset | pd.DataFrame | list | str | int | float | bool,
        roles: ABCRole | dict[str, ABCRole],
    ) -> Dataset:
        # Convert data based on its type
        if isinstance(data, dict):
            return DatasetAdapter.dict_to_dataset(data, roles)
        elif isinstance(data, pd.DataFrame):
            if isinstance(roles, ABCRole):
                raise InvalidArgumentError("roles", "dict[str, ABCRole]")
            return DatasetAdapter.frame_to_dataset(data, roles)
        elif isinstance(data, list):
            if isinstance(roles, ABCRole):
                raise InvalidArgumentError("roles", "dict[str, ABCRole]")
            return DatasetAdapter.list_to_dataset(data, roles)
        elif isinstance(data, np.ndarray):
            return DatasetAdapter.ndarray_to_dataset(data, roles)
        elif any(isinstance(data, t) for t in [str, int, float, bool]):
            return DatasetAdapter.value_to_dataset(data, roles)
        elif isinstance(data, Dataset):
            return data
        else:
            raise InvalidArgumentError("data", "dict, pd.DataFrame, list, Dataset")

    @staticmethod
    def value_to_dataset(
        data: ScalarType, roles: ABCRole | dict[str, ABCRole]
    ) -> Dataset:
        if isinstance(roles, ABCRole):
            roles = {"value": roles}
        return Dataset(
            roles=roles, data=pd.DataFrame({next(iter(roles.keys())): [data]})
        )

    @staticmethod
    def dict_to_dataset(data: dict, roles: ABCRole | dict[str, ABCRole]) -> Dataset:
        roles_names = list(data.keys())
        if any(
            [
                any(isinstance(i, t) for t in [int, str, float, bool])
                for i in list(data.values())
            ]
        ):
            data = [data]
        if isinstance(roles, dict):
            return Dataset.from_dict(data=data, roles=roles)
        elif isinstance(roles, ABCRole):
            return Dataset.from_dict(
                data=data, roles={name: roles for name in roles_names}
            )

    @staticmethod
    def list_to_dataset(data: list, roles: dict[str, ABCRole]) -> Dataset:
        return Dataset(
            roles= roles if len(roles) > 0 else {0: DefaultRole()},
            data=pd.DataFrame(data=data, columns=[next(iter(roles.keys()))] if len(roles) > 0 else [0]),
        )

    @staticmethod
    def frame_to_dataset(data: pd.DataFrame, roles: dict[str, ABCRole]) -> Dataset:
        return Dataset(
            roles=roles,
            data=data,
        )
    
    @staticmethod
    def ndarray_to_dataset(data: np.ndarray, roles: dict[str, ABCRole]) -> Dataset:
        columns = range(data.shape[1]) if len(roles) == 0 else list(roles.keys())
        data = pd.DataFrame(data=data, columns=columns)
        return Dataset(
            roles=roles,
            data=data,
        )
