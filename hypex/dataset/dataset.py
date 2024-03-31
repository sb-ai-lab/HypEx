import json
import warnings
from typing import Union, List, Iterable, Any, Type, Dict

import pandas as pd

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import StatisticRole, InfoRole, ABCRole
from hypex.dataset.utils import parse_roles
from hypex.errors.errors import (
    RoleColumnError,
    ConcatDataError,
    ConcatBackendError,
    SpaceError,
)
from hypex.utils.enums import ExperimentDataEnum
from hypex.utils.typings import FromDictType
from hypex.utils.constants import ID_SPLIT_SYMBOL


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

    def set_data(
        self,
        data: Union[pd.DataFrame, str, Type, None] = None,
        roles: Union[
            Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole], None
        ] = None,
        backend: Union[str, None] = None,
    ):
        self._backend = (
            self._select_backend_from_str(data, backend)
            if backend
            else self._select_backend_from_data(data)
        )
        if not roles:
            roles = {}
            warnings.warn(
                "Didn't add roles for experiment estimation. "
                "This fact can lead to incorrect operation of the pipeline. "
            )
        roles = (
            parse_roles(roles)
            if any(isinstance(role, type) for role in roles.keys())
            else roles
        )
        if (
            roles
            and data is not None
            and any(i not in self._backend.columns for i in list(roles.keys()))
        ):
            raise RoleColumnError(list(roles.keys()), self._backend.columns)
        self.roles = roles
        self.data = self._backend.data
        self.loc = self.Locker(self._backend, self.roles)
        self.iloc = self.ILocker(self._backend, self.roles)

    @staticmethod
    def _select_backend_from_data(data):
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        if backend == "pandas":
            return PandasDataset(data)
        return PandasDataset(data)

    def __init__(
        self,
        data: Union[pd.DataFrame, str, None] = None,
        roles: Union[
            Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]], None
        ] = None,
        backend: Union[str, None] = None,
    ):
        self.roles = None
        self.tmp_roles: Union[
            Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]]
        ] = {}
        self._backend = None
        self.data = None
        self.loc = None
        self.iloc = None
        self.set_data(data, roles, backend)

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self._backend.__len__()

    def __getitem__(self, item: Union[Iterable, str, int]):
        items = [item] if isinstance(item, str) or not isinstance(item, Iterable) else item
        roles: Dict = {
            column: self.roles[column]
            if column in self.columns and self.roles.get(column, 0)
            else InfoRole()
            for column in items
        }
        result = Dataset(data=self._backend.__getitem__(item), roles=roles)
        result.tmp_roles = self.tmp_roles
        return result

    def __setitem__(self, key: str, value: Any):
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
    ) -> List[Union[str, ABCRole]]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        roles_for_search = self.tmp_roles if tmp_role else self.roles
        return [
            column
            for column, role in roles_for_search.items()
            if any(isinstance(r, role.__class__) for r in roles)
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

    def from_dict(self, data: FromDictType, index=None):
        self._backend = self._backend.from_dict(data, index)
        if self.roles and any(
            i not in self._backend.columns for i in list(self.roles.keys())
        ):
            raise RoleColumnError(list(self.roles.keys()), self._backend.columns)
        self.data = self._backend.data
        return self

    def to_dict(self):
        return {
            "backend": str(self._backend.__class__.__name__).lower()[:-7],
            "roles": {
                "role_names": list(map(lambda x: x.role_name, list(self.roles.keys()))),
                "columns": list(self.roles.values()),
            },
            "data": self._backend.to_dict(),
        }

    def to_json(self, filename: Union[str, None] = None):
        if not filename:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    # TODO add roles
    def apply(self, func, axis=0, **kwargs):
        return Dataset(data=self._backend.apply(func=func, axis=axis, **kwargs))

    # TODO add roles
    def map(self, func, na_action=None, **kwargs):
        return self._backend.map(func=func, na_action=na_action)

    # TODO add roles
    def unique(self):
        return Dataset(data=self._backend.unique())

    # TODO add roles
    def isin(self, values: Iterable):
        return Dataset(data=self._backend.isin(values))

    def groupby(
        self,
        by: Union[str, List],
        axis: int = 0,
        level=None,
        func: Union[str, None] = None,
        fields_list: Union[List, str, None] = None,
    ):
        datasets = [
            (i, Dataset(data=data, roles=self.roles))
            for i, data in self._backend.groupby(by=by, axis=axis, level=level)
        ]
        if func:
            if fields_list:
                fields_list = (
                    fields_list if isinstance(fields_list, Iterable) else [fields_list]
                )
                datasets = [
                    (i, Dataset(data=data[fields_list].agg(func).data, roles={k: v for k, v in self.roles.items() if k in fields_list}))
                    for i, data in datasets
                ]
            else:
                datasets = [
                    (i, Dataset(data=data.loc[:, :].agg(func).data, roles=self.roles))
                    for i, data in datasets
                ]
        for i in range(len(datasets)):
            datasets[i][1].temp_roles = self.tmp_roles
        return datasets

    # TODO add roles
    def mean(self):
        return Dataset(data=self._backend.mean())

    # TODO add roles
    def max(self):
        return Dataset(data=self._backend.max())

    # TODO add roles
    def min(self):
        return Dataset(data=self._backend.min())

    # TODO add roles
    def count(self):
        return Dataset(data=self._backend.count())

    # TODO add roles
    def sum(self):
        return Dataset(data=self._backend.sum())

    # TODO add roles
    def agg(self, func: Union[str, List]):
        return Dataset(data=self._backend.agg(func))


class ExperimentData(Dataset):
    def __init__(self, data: Dataset):
        self.additional_fields = Dataset(data.data)._create_empty(index=data.index)
        self.stats_fields = Dataset(data.data)._create_empty(index=data.columns)
        self.additional_fields = Dataset(data.data)._create_empty(index=data.index)
        self.analysis_tables: Dict[str, Dataset] = {}
        self._id_name_mapping: Dict[str, str] = {}

        super().__init__(data=data.data, roles=data.roles)

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

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: str,
        name: str,
        value: Any,
        key: Union[str, None] = None,
        role=None,
    ):
        if space == ExperimentDataEnum.additional_fields:
            self.additional_fields.add_column(data=value, name=executor_id, role=role)
        elif space == ExperimentDataEnum.analysis_tables:
            self.analysis_tables[executor_id] = value
        elif space == ExperimentDataEnum.stats_fields:
            if executor_id not in self.stats_fields.columns:
                self.stats_fields.add_column(
                    data=[None] * len(self.stats_fields),
                    name=executor_id,
                    role=StatisticRole(),
                )
            self.stats_fields[executor_id][key] = value
        self._id_name_mapping[executor_id] = name
        return self

    def get_ids(
        self, classes: Union[type, List[type]]
    ) -> Dict[type, Dict[str, List[str]]]:
        classes = classes if isinstance(classes, Iterable) else [classes]
        return {
            c: {
                "stats": [
                    str(_id)
                    for _id in self.stats_fields.columns
                    if _id.split(ID_SPLIT_SYMBOL)[0] == c.__name__
                ],
                "additional_fields": [
                    str(_id)
                    for _id in self.additional_fields.columns
                    if _id.split(ID_SPLIT_SYMBOL)[0] == c.__name__
                ],
                "analysis_tables": [
                    str(_id)
                    for _id in self.analysis_tables
                    if _id.split(ID_SPLIT_SYMBOL)[0] == c.__name__
                ],
            }
            for c in classes
        }
