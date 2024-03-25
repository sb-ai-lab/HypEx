import json
import warnings
from typing import Dict, Union, List, Iterable, Any, Type

import pandas as pd

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import ABCRole, StatisticRole, InfoRole
from hypex.dataset.utils import parse_roles
from hypex.errors.errors import (
    RoleColumnError,
    ConcatDataError,
    ConcatBackendError,
    SpaceError,
)
from hypex.utils.hypex_enums import ExperimentDataEnum, BackendsEnum
from hypex.utils.hypex_typings import RolesType


class Dataset(DatasetBase):
    class Locker:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, item):
            return Dataset(data=self.backend.loc(item))

    class ILocker:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, item):
            return Dataset(data=self.backend.iloc(item))

    def set_data(
        self,
        data: Union[pd.DataFrame, str, Type],
        roles: RolesType = None,
        backend: str = None,
    ):
        self._backend = (
            self._select_backend_from_str(data, backend)
            if backend
            else self._select_backend_from_data(data)
        )
        # TODO check with two variant
        if roles and any(i not in self._backend.columns for i in list(roles.values())):
            raise RoleColumnError(list(roles.keys()), self._backend.columns)
        self.roles = parse_roles(roles)
        self.data = self._backend.data
        self.loc = self.Locker(self._backend)
        self.iloc = self.ILocker(self._backend)

    @staticmethod
    def _select_backend_from_data(data):
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        if backend == BackendsEnum.pandas:
            return PandasDataset(data)

    def __init__(
        self,
        data: Union[pd.DataFrame, str] = None,
        roles: RolesType = None,
        backend: str = None,
    ):
        self.roles = None
        self.tmp_roles: Union[None, Dict[Union[List[str], str], ABCRole]] = None
        self._backend = None
        self.data = None
        self.loc = None
        self.iloc = None
        self.set_data(data, roles, backend)

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self._backend.__len__()

    def __getitem__(self, item):
        items = item if isinstance(item, Iterable) else [item]
        roles = {}
        for column in items:
            if column in self.columns and self.roles.get(column, 0):
                roles[self.roles[column]] = column
            else:
                roles[InfoRole] = column
        return Dataset(data=self._backend.__getitem__(item), roles=roles)

    def __setitem__(self, key, value):
        if key not in self.columns and isinstance(key, str):
            self.add_column(value, key, InfoRole())
            warnings.warn("Column must be added by add_column", category=Warning)
        self.data[key] = value

    def _create_empty(self, index=None, columns=None):
        index = [] if index is None else index
        columns = [] if columns is None else columns
        self._backend = self._backend._create_empty(index, columns)
        self.data = self._backend.data
        return self

    def get_columns_by_roles(
        self, roles: Union[ABCRole, Iterable[ABCRole]], tmp_role=False
    ) -> List[str]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        get_roles = self.roles if not tmp_role else self.tmp_roles
        return [
            column
            for column, role in get_roles.items()
            if any(isinstance(role, r) for r in roles)
        ]

    @property
    def index(self):
        return self._backend.index

    @property
    def columns(self):
        return self._backend.columns

    def add_column(self, data, name: Union[str, int, List], role: ABCRole, index=None):
        self.roles.update({name: role})
        if isinstance(data, Dataset):
            data = data._backend.data[list(data._backend.data.columns)[0]]
        self._backend.add_column(data, name, index)

    def append(self, other, index=None):
        if not isinstance(other, Dataset):
            raise ConcatDataError(type(other))
        if type(other._backend) != type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))
        return Dataset(data=self._backend.append(other._backend, index))

    def from_dict(self, data, index=None):
        self._backend = self._backend.from_dict(data, index)
        self.data = self._backend.data
        return self

    def to_dict(self):
        return {
            "backend": str(self._backend.__class__.__name__).lower()[:-7],
            "roles": {
                "role_names": list(
                    map(lambda x: x._role_name, list(self.roles.keys()))
                ),
                "columns": list(self.roles.values()),
            },
            "data": self._backend.to_dict(),
        }

    def to_json(self, filename: str = None):
        if filename is None:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    def apply(self, func, axis=0, **kwargs):
        return Dataset(data=self._backend.apply(func=func, axis=axis, **kwargs))

    def map(self, func, na_action=None, **kwargs):
        return self._backend.map(func=func, na_action=na_action)

    def unique(self):
        return self._backend.unique()

    def isin(self, values: Iterable):
        return Dataset(data=self._backend.isin(values))

    def groupby(
        self,
        by: Union[str, List],
        axis: int = 0,
        level=None,
        func: str = None,
        fields_list: List = None,
    ):
        datasets = [
            (i[0], Dataset(data=i[1]))
            for i in self._backend.groupby(by=by, axis=axis, level=level)
        ]
        if func:
            if fields_list:
                datasets = [
                    (i[0], Dataset(data=i[1][fields_list].agg(func).data))
                    for i in datasets
                ]
            else:
                datasets = [
                    (i[0], Dataset(data=i[1].loc[:, :].agg(func).data))
                    for i in datasets
                ]
        return iter(datasets)

    def mean(self):
        return self._backend.mean()

    def max(self):
        return self._backend.max()

    def min(self):
        return self._backend.min()

    def count(self):
        return self._backend.count()

    def sum(self):
        return self._backend.sum()

    def agg(self, func: Union[str, List]):
        return Dataset(data=self._backend.agg(func))


class ExperimentData(Dataset):
    def __init__(self, data: Any):
        super().__init__(data)
        if isinstance(data, Dataset):
            self.additional_fields = Dataset(data.data)._create_empty(index=data.index)
            self.stats_fields = Dataset(data.data)._create_empty(index=data.columns)
            self.additional_fields = Dataset(data.data)._create_empty(index=data.index)
            self.stats_fields = Dataset(data.data)._create_empty(index=data.columns)
        else:
            data = Dataset(data)
            self.additional_fields = data
            self.stats_fields = data

        self.data = data
        self.analysis_tables = {}
        self._id_name_mapping = {}

    def _create_empty(self, index=None, columns=None):
        self.additional_fields._create_empty(index, columns)
        self.stats_fields._create_empty(index, columns)
        return self

    def check_hash(self, executor_id: int, space: str) -> bool:
        if space == ExperimentDataEnum.additional_fields:
            return executor_id in self.additional_fields.columns
        elif space == ExperimentDataEnum.stats_fields:
            return executor_id in self.stats_fields.columns
        elif space == ExperimentDataEnum.analysis_tables:
            return executor_id in self.analysis_tables
        else:
            raise SpaceError(space)

    def _create_empty(self, indexes=None, columns=None):
        self.additional_fields._create_empty(indexes, columns)
        self.stats_fields._create_empty(indexes, columns)
        return self

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: int,
        name: str,
        value: Any,
        key: str = None,
        role=None,
    ):
        if space == ExperimentDataEnum.additional_fields:
            self.additional_fields.add_column(data=value, name=executor_id, role=role)
        elif space == ExperimentDataEnum.analysis_tables:
            self.analysis_tables[name] = value
        elif space == ExperimentDataEnum.stats_fields:
            if executor_id not in self.stats_fields.columns:
                self.stats_fields.add_column(
                    data=[None] * len(self.stats_fields),
                    name=executor_id,
                    role=StatisticRole(),
                )
            self.stats_fields[executor_id][key] = value
        self._id_name_mapping[executor_id] = name
