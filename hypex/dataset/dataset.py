import json
import warnings
from typing import Union, List, Iterable, Any, Dict, Callable, Hashable, Optional

import pandas as pd  # type: ignore

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import StatisticRole, InfoRole, ABCRole, default_roles
from hypex.utils.constants import ID_SPLIT_SYMBOL
from hypex.utils.enums import ExperimentDataEnum, BackendsEnum
from hypex.utils.errors import NotFoundInExperimentDataError
from hypex.utils.errors import (
    RoleColumnError,
    ConcatDataError,
    ConcatBackendError,
)
from hypex.utils.typings import FromDictType


def parse_roles(roles: Dict) -> Dict[Union[str, int], ABCRole]:
    new_roles = {}
    roles = roles or {}
    for role in roles:
        r = default_roles.get(role, role)
        if isinstance(roles[role], list):
            for i in roles[role]:
                new_roles[i] = r
        else:
            new_roles[roles[role]] = r
    return new_roles or roles


class Dataset(DatasetBase):
    class Locker:
        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item):
            t_data = self.backend.loc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

    class ILocker:
        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item):
            t_data = self.backend.iloc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

    @staticmethod
    def _select_backend_from_data(data):
        """
        Выбирает бэкенд исходя из типа пришедших данных
        """
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        """
        Выбирает бэкенд исходя из строкового описания бэкенда
        """
        if backend == BackendsEnum.pandas:
            return PandasDataset(data)
        return PandasDataset(data)

    def __init__(
        self,
        roles: Union[
            Dict[ABCRole, Union[List[Union[str, int]], str, int]],
            Dict[Union[str, int], ABCRole],
        ],
        data: Optional[Union[pd.DataFrame, str]] = None,
        backend: Optional[BackendsEnum] = None,
    ):
        self.tmp_roles: Union[
            Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]]
        ] = {}
        self._backend = (
            self._select_backend_from_str(data, backend)
            if backend
            else self._select_backend_from_data(data)
        )
        roles = (
            parse_roles(roles)
            if any(isinstance(role, type) for role in roles.keys())
            else roles
        )
        if data is not None and any(
            i not in self._backend.columns for i in list(roles.keys())
        ):
            raise RoleColumnError(list(roles.keys()), self._backend.columns)
        self.roles: Dict[Union[str, int], ABCRole] = roles
        self.loc = self.Locker(self._backend, self.roles)
        self.iloc = self.ILocker(self._backend, self.roles)

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self._backend.__len__()

    def __getitem__(self, item: Union[Iterable, str, int]):
        items = (
            [item] if isinstance(item, str) or not isinstance(item, Iterable) else item
        )
        roles: Dict = {
            column: (
                self.roles[column]
                if column in self.columns and self.roles.get(column, 0)
                else InfoRole()
            )
            for column in items
        }
        result = Dataset(data=self._backend.__getitem__(item), roles=roles)
        result.tmp_roles = self.tmp_roles
        return result

    def __setitem__(self, key: str, value: Any):
        if key not in self.columns and isinstance(key, str):
            self.add_column(value, {key: InfoRole()})
            warnings.warn("Column must be added by add_column", category=SyntaxWarning)
        self.data[key] = value

    @staticmethod
    def _create_empty(
        roles: Dict[Any, ABCRole], backend=BackendsEnum.pandas, index=None
    ):
        index = [] if index is None else index
        columns = list(roles.keys())
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend._create_empty(index, columns)
        ds.data = ds._backend.data
        return ds

    def get_columns_by_roles(
        self, roles: Union[ABCRole, Iterable[ABCRole]], tmp_role=False
    ) -> List[Union[str, int]]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        roles_for_search: Dict[Union[str, int], ABCRole] = (
            self.tmp_roles if tmp_role else self.roles
        )
        return [
            column
            for column, role in roles_for_search.items()
            if any(isinstance(r, role.__class__) for r in roles)
        ]

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

    def add_column(
        self,
        data,
        role: Optional[Dict[str, ABCRole]] = None,
        index: Optional[Iterable[Hashable]] = None,
    ):
        if role is None:  # если данные - датасет
            if not isinstance(data, Dataset):
                raise ValueError("Козьёль")
            self.roles.update(data.roles)
            self._backend.add_column(
                data._backend.data[list(data._backend.data.columns)[0]],
                list(data.roles.keys())[0],
                index,
            )
        else:
            self.roles.update(role)
            self._backend.add_column(data, list(role.keys())[0], index)

    def append(self, other, index=None):
        if not isinstance(other, Dataset):
            raise ConcatDataError(type(other))
        if type(other._backend) != type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))
        self.roles.update(other.roles)
        return Dataset(
            roles=self.roles, data=self._backend.append(other._backend, index)
        )

    @staticmethod
    def from_dict(
        data: FromDictType,
        roles: Union[
            Dict[ABCRole, Union[List[Union[str, int]], str, int]],
            Dict[Union[str, int], ABCRole],
        ],
        backend: BackendsEnum = BackendsEnum.pandas,
        index=None,
    ):
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend.from_dict(data, index)
        ds.data = ds._backend.data
        return ds

    def to_dict(self):
        return {
            "backend": self._backend.name,
            "roles": {
                "role_names": list(map(lambda x: x, list(self.roles.keys()))),
                "columns": list(self.roles.values()),
            },
            "data": self._backend.to_dict(),
        }

    def to_json(self, filename: Optional[str] = None):
        if not filename:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    def apply(
        self,
        func: Callable,
        role: Dict[Union[str, int], ABCRole],
        axis=0,
        **kwargs,
    ):
        return Dataset(
            data=self._backend.apply(func=func, axis=axis, **kwargs).rename(
                list(role.keys())[0]
            ),
            roles=role,
        )

    def map(self, func, na_action=None, **kwargs):
        return self._backend.map(func=func, na_action=na_action)

    def unique(self):
        return Dataset(data=self._backend.unique())

    def isin(self, values: Iterable):
        return Dataset(roles=self.roles, data=self._backend.isin(values))

    def groupby(
        self,
        by: Any,
        func: Optional[Union[str, List]] = None,
        fields_list: Optional[Union[str, List]] = None,
        **kwargs,
    ):
        datasets = [
            (i, Dataset(roles=self.roles, data=data))
            for i, data in self._backend.groupby(by=by, **kwargs)
        ]
        if func:
            if fields_list:
                fields_list = (
                    fields_list if isinstance(fields_list, Iterable) else [fields_list]
                )
                datasets = [
                    (
                        i,
                        Dataset(
                            roles={
                                k: v for k, v in self.roles.items() if k in fields_list
                            },
                            data=data[fields_list].agg(func).data,
                        ),
                    )
                    for i, data in datasets
                ]
            else:
                datasets = [
                    (i, Dataset(roles=self.roles, data=data.agg(func).data))
                    for i, data in datasets
                ]
        for dataset in datasets:
            dataset[1].tmp_roles = self.tmp_roles
        return datasets

    def mean(self):
        return Dataset(data=self._backend.mean(), roles=self.roles)

    def max(self):
        return Dataset(data=self._backend.max(), roles=self.roles)

    def min(self):
        return Dataset(data=self._backend.min(), roles=self.roles)

    def count(self):
        return Dataset(data=self._backend.count(), roles=self.roles)

    def sum(self):
        return Dataset(data=self._backend.sum(), roles=self.roles)

    def agg(self, func: Union[str, List]):
        func = func if isinstance(func, List) else [func]
        return Dataset(data=self._backend.agg(func), roles=self.roles)


class ExperimentData(Dataset):
    def __init__(self, data: Dataset):
        self.additional_fields = Dataset._create_empty(roles={}, index=data.index)
        self.stats = Dataset._create_empty(roles={}, index=data.columns)
        self.additional_fields = Dataset._create_empty(roles={}, index=data.index)
        self.analysis_tables: Dict[str, Dataset] = {}
        self.id_name_mapping: Dict[str, str] = {}

        super().__init__(data=data.data, roles=data.roles)

    @staticmethod
    def _create_empty(roles: Dict[Any, ABCRole], backend="pandas", index=None):
        ds = Dataset._create_empty(roles, backend, index)
        return ExperimentData(ds)

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
        if space == ExperimentDataEnum.additional_fields:
            return executor_id in self.additional_fields.columns
        elif space == ExperimentDataEnum.stats:
            return executor_id in self.stats.columns
        elif space == ExperimentDataEnum.analysis_tables:
            return executor_id in self.analysis_tables
        else:
            return any(self.check_hash(executor_id, s) for s in ExperimentDataEnum)

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: str,
        name: str,
        value: Any,
        key: Optional[str] = None,
        role=None,
    ):
        if space == ExperimentDataEnum.additional_fields:
            self.additional_fields.add_column(data=value, role={executor_id: role})
        elif space == ExperimentDataEnum.analysis_tables:
            self.analysis_tables[executor_id] = value
        elif space == ExperimentDataEnum.stats:
            if executor_id not in self.stats.columns:
                self.stats.add_column(
                    data=[None] * len(self.stats),
                    role={executor_id: StatisticRole()},
                )
            self.stats[executor_id][key] = value
        self.id_name_mapping[executor_id] = name
        return self

    def get_ids_by_executors(
        self, classes: Union[type, List[type]]
    ) -> Dict[type, Dict[str, List[str]]]:
        classes = classes if isinstance(classes, Iterable) else [classes]
        return {
            class_: {
                ExperimentDataEnum.stats.value: [
                    str(_id)
                    for _id in self.stats.columns
                    if _id.split(ID_SPLIT_SYMBOL)[0] == class_.__name__
                ],
                ExperimentDataEnum.additional_fields.value: [
                    str(_id)
                    for _id in self.additional_fields.columns
                    if _id.split(ID_SPLIT_SYMBOL)[0] == class_.__name__
                ],
                ExperimentDataEnum.analysis_tables.value: [
                    str(_id)
                    for _id in self.analysis_tables
                    if _id.split(ID_SPLIT_SYMBOL)[0] == class_.__name__
                ],
            }
            for class_ in classes
        }

    def _get_one_id(self, class_: type, space: ExperimentDataEnum) -> str:
        result = self.get_ids_by_executors(class_)
        if not len(result):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

    def get_ids_by_field(self):
        pass
