from __future__ import annotations

import warnings
import copy
from copy import deepcopy
import json  # type: ignore
from abc import ABC
from typing import Any, Iterable, Callable, Hashable, Literal, Optional, Sequence

import pandas as pd  # type: ignore
from numpy import ndarray
import pyspark.sql as spark

from ..utils import (
    BackendsEnum,
    BackendTypeError,
    ConcatBackendError,
    ConcatDataError,
    DataTypeError,
    ScalarType,
    RoleColumnError,
    SourceDataTypes,
)
from ..utils.adapter import Adapter
from .backends import PandasDataset, SparkDataset
from .roles import (
    ABCRole,
    DefaultRole,
    FilterRole,
    InfoRole,
    StatisticRole,
    default_roles,
)


class DatasetBase(ABC):
    @staticmethod
    def _select_backend_from_data(data, session: spark.SparkSession = None):
        if isinstance(data, pd.DataFrame):
            return PandasDataset(data)
        elif isinstance(data, spark.DataFrame):
            return SparkDataset(data, session)
        raise TypeError(
            "Data must be an instance of either pandas.DataFrame or spark.DataFrame"
        )

    @staticmethod
    def _select_backend_from_str(data, backend, session=None):
        if backend == BackendsEnum.pandas:
            return PandasDataset(data)
        elif backend == BackendsEnum.spark:
            return SparkDataset(data, session)
        elif backend is None:
            if session is not None:
                return SparkDataset(data, session)
            return PandasDataset(data)
        raise TypeError("Backend must be an instance of BackendsEnum")

    def _set_all_roles(self, roles):
        keys = list(roles.keys())
        for column in self.columns:
            if column not in keys:
                roles[column] = copy.deepcopy(self.default_role) or DefaultRole()
        return roles

    def _set_empty_types(self, roles):
        colunms_dtypes = self._backend.get_column_type(self._backend.columns)
        new_types = {}
        for column, role in roles.items():
            if role.data_type is None:
                role.data_type = colunms_dtypes[column]
            elif role.data_type != colunms_dtypes[column]:
                new_types[column] = role.data_type
        self._backend = self._backend.update_column_type(new_types)

    def __init__(
        self,
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole],
        data: spark.DataFame | pd.DataFrame | str | None = None,
        backend: BackendsEnum | None = None,
        default_role: ABCRole | None = None,
        session: Optional[spark.SparkSession] = None,
    ):
        if backend is not None:
            self._backend = self._select_backend_from_str(data, backend, session)
        elif any(
            isinstance(data, source_data_type)
            for source_data_type in [
                *SourceDataTypes.__args__,
                PandasDataset,
                SparkDataset,
                self.__class__,
            ]
        ):
            self._backend = self._select_backend_from_data(data, session)
        else:
            if session is not None:
                self._backend = SparkDataset(data, session)
            self._backend = PandasDataset(data)

        self.default_role = default_role
        if roles is None and data.hasattr("roles") and data.roles is not None:
            roles = data.roles
        else:
            roles = (
                self._parse_roles(roles)
                if any(isinstance(role, ABCRole) for role in roles.keys())
                else roles
            )
            if any(not isinstance(role, ABCRole) for role in roles.values()):
                raise TypeError("Roles must be instances of ABCRole type")
            if data is not None and any(
                i not in self._backend.columns for i in list(roles.keys())
            ):
                raise RoleColumnError(list(roles.keys()), self._backend.columns)
        if data is not None:
            roles = self._set_all_roles(roles)
            self._set_empty_types(roles)
        self._roles: dict[str, ABCRole] = roles
        self._tmp_roles: (
            dict[ABCRole, list[str] | str] | dict[list[str] | str] | ABCRole
        ) = {}

    def __repr__(self):
        return self._backend.__repr__()

    def _repr_html_(self):
        return self._backend._repr_html_()

    def __len__(self):
        return self._backend.__len__()

    def __getitem__(self, item: Iterable | str | int) -> DatasetBase:
        if isinstance(item, DatasetBase):
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
        result = self.__class__(data=self._backend.__getitem__(item), roles=roles)
        result.tmp_roles = {
            key: value for key, value in self.tmp_roles.items() if key in items
        }
        return result

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, DatasetBase):
            value = value.iselect(0).data
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

    @classmethod
    def create_empty(
        cls, roles=None, index=None, backend=BackendsEnum.pandas
    ) -> DatasetBase:
        if roles is None:
            roles = {}
        index = [] if index is None else index
        columns = list(roles.keys())
        ds = cls(roles=roles, backend=backend)
        ds._backend = ds._backend.create_empty(index, columns)
        ds.data = ds.backend.data
        return ds

    @staticmethod
    def _parse_roles(roles: dict) -> dict[str | int] | ABCRole:
        new_roles = {}
        roles = roles or {}
        for role in roles:
            r = default_roles.get(role, role)
            if isinstance(roles[role], list):
                for i in roles[role]:
                    new_roles[i] = copy.deepcopy(r)
            else:
                new_roles[roles[role]] = copy.deepcopy(r)
        return new_roles or roles

    def get(
        self,
        key,
        default=None,
    ) -> DatasetBase:
        return self.__class__(
            data=self._backend.get(key, default), roles=deepcopy(self.roles)
        )

    def take(
        self,
        indices: int | list[int],
        axis: Literal["index", "columns", "rows"] | int = 0,
    ) -> DatasetBase:
        new_data = self._backend.take(indices=indices, axis=axis)
        new_roles = (
            {k: deepcopy(v) for k, v in self.roles.items() if k in new_data.columns}
            if axis == 1
            else deepcopy(self.roles)
        )
        return self.__class__(data=new_data, roles=new_roles)

    def __binary_magic_operator(self, other, func_name: str) -> Any:
        if not any(
            isinstance(other, t)
            for t in [self.__class__, str, int, float, bool, Sequence]
        ):
            raise DataTypeError(type(other))
        func = getattr(self._backend, func_name)
        t_roles = deepcopy(self.roles)
        for role in t_roles.values():
            role.data_type = None
        if isinstance(other, self.__class__):
            if type(other._backend) is not type(self._backend):
                raise BackendTypeError(type(other._backend), type(self._backend))
            other = other.rename(
                {
                    other.columns[i]: self.data.columns[i]
                    for i in range(len(other.columns))
                }
            ).backend
        return self.__class__(roles=t_roles, data=func(other))

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
        return self.__class__(roles=self.roles, data=(+self._backend))

    def __neg__(self):
        return self.__class__(roles=self.roles, data=(-self._backend))

    def __abs__(self):
        return self.__class__(roles=self.roles, data=abs(self._backend))

    def __invert__(self):
        return self.__class__(roles=self.roles, data=(~self._backend))

    def __round__(self, ndigits: int = 0):
        return self.__class__(roles=self.roles, data=round(self._backend, ndigits))

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

    def search_columns(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role=False,
        search_types: list | None = None,
    ) -> list[str]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        roles_for_search = self._tmp_roles if tmp_role else self.roles
        return [
            str(column)
            for column, role in roles_for_search.items()
            if any(
                isinstance(r, role.__class__)
                and (not search_types or role.data_type in search_types)
                for r in roles
            )
        ]

    def search_columns_by_type(
        self,
        search_types: list | type,
    ) -> list[str]:
        search_types = (
            search_types if isinstance(search_types, Iterable) else [search_types]
        )
        return [
            str(column)
            for column, role in self.roles.items()
            if any(role.data_type == t for t in search_types)
        ]

    def replace_roles(
        self,
        new_roles_map: dict[ABCRole | str] | ABCRole,
        tmp_role: bool = False,
        auto_roles_types: bool = False,
    ):
        new_roles_map = self._parse_roles(
            {
                role: (
                    self.search_columns(column, tmp_role)
                    if isinstance(column, ABCRole)
                    else column
                )
                for column, role in new_roles_map.items()
            }
        )

        new_roles = {
            column: new_roles_map[column] if column in new_roles_map else role
            for column, role in self.roles.items()
        }

        if tmp_role:
            self._tmp_roles = new_roles
        else:
            self.roles = new_roles
            if auto_roles_types:
                self._set_empty_types(new_roles_map)

        return self

    @property
    def index(self):
        return self._backend.index

    @property
    def data(self):
        return self._backend.data

    @data.setter
    def data(self, value):
        self._backend.data = value

    @property
    def columns(self):
        return self._backend.columns

    @property
    def roles(self):
        return self._roles

    @roles.setter
    def roles(self, value):
        self._set_roles(new_roles_map=value, temp_role=False)

    @property
    def shape(self):
        return self._backend.shape

    @property
    def tmp_roles(self):
        return self._tmp_roles

    @property
    def session(self):
        return self._backend.session

    @tmp_roles.setter
    def tmp_roles(self, value):
        self._set_roles(new_roles_map=value, temp_role=True)
        self._set_empty_types(self._tmp_roles)

    def _convert_data_after_agg(self, result) -> DatasetBase | float:
        if isinstance(result, ScalarType):
            return result
        role: ABCRole = StatisticRole()
        return self.__class__(
            data=result, roles={column: role for column in result.columns}
        )

    def add_column(
        self,
        data,
        role: dict[str, ABCRole] | None = None,
        index: Iterable[Hashable] | None = None,
    ):
        if role is None:
            if not isinstance(data, self.__class__):
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
            if isinstance(data, self.__class__):
                data = data.data
            self.roles.update(role)
            self._backend.add_column(data, list(role.keys()), index)
        return self

    def _check_other_dataset(self, other):
        if not isinstance(other, self.__class__):
            raise ConcatDataError(type(other))
        if type(other._backend) is not type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))

    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> DatasetBase:
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

        return self.__class__(roles=new_roles, data=new_backend.data)

    def append(self, other, axis=0) -> DatasetBase:
        other = Adapter.to_list(other)

        new_roles = deepcopy(self.roles)
        for o in other:
            self._check_other_dataset(o)
            new_roles.update(o.roles)

        return self.__class__(roles=new_roles, data=self.backend.append(other, axis))

    def apply(
        self,
        func: Callable,
        role: dict[str, ABCRole],
        axis: int = 0,
        **kwargs,
    ) -> DatasetBase:
        if self.is_empty():
            return deepcopy(self)
        tmp_data = self._backend.apply(
            func=func, axis=axis, column_name=next(iter(role.keys())), **kwargs
        )
        # tmp_roles = (
        #     {next(iter(role.keys())): next(iter(role.values()))}
        #     if ((Dataset({}, tmp_data).is_empty()) and len(role) > 1)
        #     else role
        # )
        tmp_roles = deepcopy(role)
        return self.__class__(
            data=tmp_data,
            roles=tmp_roles,
        )

    def map(self, func, na_action=None, **kwargs) -> DatasetBase:
        return self.__class__(
            roles=self.roles,
            data=self._backend.map(func=func, na_action=na_action, **kwargs),
        )

    def is_empty(self) -> bool:
        return self._backend.is_empty()

    def unique(self) -> dict[str, list[Any]]:
        return self._backend.unique()

    def nunique(self, dropna: bool = False) -> dict[str, int]:
        return self._backend.nunique(dropna)

    def isin(self, values: Iterable) -> DatasetBase:
        role: ABCRole = FilterRole()
        return self.__class__(
            roles={column: role for column in self.roles.keys()},
            data=self._backend.isin(values),
        )

    def groupby(
        self,
        by: Any,
        func: str | list | None = None,
        fields_list: str | list | None = None,
        **kwargs,
    ) -> list[tuple[str, DatasetBase]]:
        if isinstance(by, self.__class__) and len(by.columns) == 1:
            datasets = [
                (
                    group,
                    self.__class__(
                        roles=self.roles, data=self.data.loc[group_data.index]
                    ),
                )
                for group, group_data in by._backend.groupby(by=by.columns[0], **kwargs)
            ]
        else:
            datasets = [
                (group, self.__class__(roles=self.roles, data=data))
                for group, data in self._backend.groupby(by=by, **kwargs)
            ]
        if fields_list:
            fields_list = Adapter.to_list(fields_list)
            datasets = [(i, data.select(fields_list)) for i, data in datasets]
        if func:
            datasets = [(i, data.agg(func)) for i, data in datasets]
        for dataset in datasets:
            if isinstance(dataset, self.__class__):
                dataset[1].tmp_roles = self.tmp_roles
        return datasets

    def fillna(
        self,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        **kwargs,
    ):
        if values is None and method is None:
            raise ValueError("Value or filling method must be provided")
        return self.__class__(
            roles=self.roles,
            data=self.backend.fillna(values=values, method=method, **kwargs),
        )

    def mean(self):
        return self._convert_data_after_agg(self._backend.mean())

    def max(self):
        return self._convert_data_after_agg(self._backend.max())

    def min(self):
        return self._convert_data_after_agg(self._backend.min())

    def count(self):
        if self.is_empty():
            return self.create_empty({role: InfoRole() for role in self.roles})
        return self._convert_data_after_agg(self._backend.count())

    def sum(self):
        return self._convert_data_after_agg(self._backend.sum())

    def log(self):
        return self._convert_data_after_agg(self._backend.log())

    def mode(self, numeric_only: bool = False, dropna: bool = True):
        t_data = self._backend.mode(numeric_only=numeric_only, dropna=dropna)
        return self.__class__(
            data=t_data, roles={role: InfoRole() for role in t_data.columns}
        )

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

    def corr(self, numeric_only=False):
        t_data = self._backend.corr(numeric_only=numeric_only)
        t_roles = {column: self.roles[column] for column in t_data.columns}
        return self.__class__(roles=t_roles, data=t_data)

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
        return self.__class__(roles=t_roles, data=t_data)

    def na_counts(self):
        """Count NA values"""
        return self._convert_data_after_agg(self._backend.na_counts())

    def isna(self):
        return self._convert_data_after_agg(self._backend.isna())

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
        return self.__class__(
            roles=new_roles,
            data=new_data,
        )

    def drop(
        self,
        labels: str | None = None,
        axis: int | None = None,
        columns: str | Iterable[str] | None = None,
    ):
        raise NotImplemented(
            "The method 'drop' is not implemented for this type of Dataset"
        )

    def filter(
        self,
        items: list | None = None,
        regex: str | None = None,
        axis: int | None = None,
    ) -> DatasetBase:
        t_data = self._backend.filter(items=items, regex=regex, axis=axis)
        t_roles = {c: self.roles[c] for c in t_data.columns if c in self.roles.keys()}
        return self.__class__(roles=t_roles, data=t_data)

    def select(self, columns: str | list[str]):
        columns = Adapter.to_list(columns)
        return self.filter(items=columns, axis=1)

    def iselect(self, columns: int | list[int]):
        columns = Adapter.to_list(columns)
        columns = [self.columns[n] for n in columns]
        return self.filter(items=columns, axis=1)

    def select_dtypes(self, include: Any = None, exclude: Any = None):
        # Filter data by dtypes
        t_data = self._backend.select_dtypes(include=include, exclude=exclude)

        # Keep only roles for remaining columns
        t_roles = {k: v for k, v in self.roles.items() if k in t_data.columns}
        return self.__class__(roles=t_roles, data=t_data)

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
        if not isinstance(right, self.__class__):
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
        return self.__class__(roles=new_roles, data=t_data)

    def dot(self, other: DatasetBase | ndarray) -> DatasetBase:
        return self.__class__(
            roles=deepcopy(other.roles) if isinstance(other, self.__class__) else {},
            data=self.backend.dot(
                other.backend if isinstance(other, self.__class__) else other
            ),
        )

    def sample(
        self,
        frac: float | None = None,
        n: int | None = None,
        random_state: int | None = None,
    ) -> DatasetBase:
        return self.__class__(
            self.roles,
            data=self.backend.sample(frac=frac, n=n, random_state=random_state),
        )

    def cov(self):
        t_data = self.backend.cov()
        return self.__class__(
            {column: DefaultRole() for column in t_data.columns}, data=t_data
        )

    def rename(self, names: dict[str, str]):
        roles = {names.get(column, column): role for column, role in self.roles.items()}
        return self.__class__(roles, data=self.backend.rename(names))

    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        regex: bool = False,
    ) -> DatasetBase:
        return self.__class__(
            self.roles,
            data=self._backend.replace(to_replace=to_replace, value=value, regex=regex),
        )

    def list_to_columns(self, column: str) -> DatasetBase:
        if not pd.api.types.is_list_like(self.backend[column][0]):
            return self
        extended_data = self.backend.list_to_columns(column)
        extended_roles = {
            c: deepcopy(self.roles[column]) for c in extended_data.columns
        }
        extended_ds = self.__class__(roles=extended_roles, data=extended_data)
        return self.append(extended_ds, axis=1).drop(column, axis=1)

    def to_dict(self):
        return {
            "backend": self._backend.name,
            "roles": {
                "role_names": list(map(lambda x: x, list(self.roles.keys()))),
                "columns": list(self.roles.values()),
            },
            "data": self._backend.to_dict(),
        }

    def to_numpy(self):
        return self._backend.to_numpy()

    def to_records(self):
        return self._backend.to_records()

    def to_json(self, filename: str | None = None):
        if not filename:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    @property
    def backend(self):
        return self._backend

    def get_values(
        self,
        row: str | None = None,
        column: str | None = None,
    ) -> Any:
        return self._backend.get_values(row=row, column=column)

    def iget_values(
        self,
        row: int | None = None,
        column: int | None = None,
    ) -> Any:
        return self._backend.iget_values(row=row, column=column)

    def _set_roles(
        self,
        new_roles_map: dict[ABCRole, list[str] | str] | dict[list[str] | str] | ABCRole,
        temp_role: bool = False,
    ):
        if not new_roles_map:
            if not temp_role:
                return self.roles
            else:
                self._tmp_roles = {}
                return self

        keys, values = list(new_roles_map.keys()), list(new_roles_map.values())
        roles, columns_sets = (
            (keys, values) if isinstance(keys[0], ABCRole) else (values, keys)
        )

        new_roles = {}
        for role, columns in zip(roles, columns_sets):
            if isinstance(columns, list):
                for column in columns:
                    new_roles[column] = copy.deepcopy(role)
            else:
                new_roles[columns] = copy.deepcopy(role)

        if temp_role:
            self._tmp_roles = new_roles
        else:
            self._roles = new_roles

        return self
