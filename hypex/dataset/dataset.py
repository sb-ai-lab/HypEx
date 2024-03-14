import json
import warnings
from typing import Dict, Optional, Union, List, Iterable, Any, Type

import pandas as pd

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import ABCRole, StatisticRole
from hypex.dataset.utils import parse_roles


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
        roles: Union[Dict] = None,
        backend: str = None,
    ):
        self._backend = (
            self._select_backend_from_data(data)
            if not backend
            else self._select_backend_from_str(data, backend)
        )

        if roles and not all(i in self._backend.columns for i in list(roles.values())):
            raise ValueError(
                "Check your roles. All of them must be names of data columns. \n"
                f"Now roles have {list(roles.keys())} values and columns have {self._backend.columns} values"
            )
        self.roles = parse_roles(roles)
        self.data = self._backend.data
        self.loc = self.Locker(self._backend)
        self.iloc = self.ILocker(self._backend)

    def __init__(
        self,
        data: Union[pd.DataFrame, str] = None,
        roles: Optional[Dict[ABCRole, Union[List[str], str]]] = None,
        backend: str = None,
    ):
        self.roles = None
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
        return self._backend.__getitem__(item)

    def __setitem__(self, key, value):
        if key not in self.columns and isinstance(key, str):
            self.add_column(value, key, StatisticRole)
            warnings.warn("Column must be added by add_column", category=Warning)
        self.data[key] = value

    @staticmethod
    def _select_backend_from_data(data):
        if isinstance(data, pd.DataFrame):
            return PandasDataset(data)
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        if backend == "pandas":
            return PandasDataset(data)

    def get_columns_by_roles(
        self, roles: Union[ABCRole, Iterable[ABCRole]]
    ) -> List[str]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        return [
            column
            for column, role in self.roles.items()
            if any(isinstance(role, r) for r in roles)
        ]

    def add_column(self, data, name: Union[str, int, List], role: ABCRole):
        self.roles.update({name: role})
        if isinstance(data, Dataset):
            data = data._backend.data[list(data._backend.data.columns)[0]]
        self._backend.add_column(data, name)

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

    def _create_empty(self, index=None, columns=None):
        index = [] if index is None else index
        columns = [] if columns is None else columns
        self._backend = self._backend._create_empty(index, columns)
        self.data = self._backend.data
        return self

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
                    (
                        i[0],
                        Dataset(data=eval("i[1][{}].{}()".format(fields_list, func))),
                    )
                    for i in datasets
                ]
            else:
                datasets = [
                    (i[0], Dataset(data=eval("i[1].loc[:, :].{}()".format(func))))
                    for i in datasets
                ]
        return iter(datasets)

    @property
    def index(self):
        return self._backend.index

    @property
    def columns(self):
        return self._backend.columns

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


class ExperimentData(Dataset):
    def __init__(self, data: Any):
        super().__init__(data)
        if isinstance(data, Dataset):
            self.additional_fields = Dataset(data.data)._create_empty(index=data.index)
            self.stats_fields = Dataset(data.data)._create_empty(index=data.columns)
            self.additional_fields = Dataset(data.data)._create_empty(index=data.index)
            self.stats_fields = Dataset(data.data)._create_empty(index=data.columns)
        else:
            self.additional_fields = Dataset(data)
            self.stats_fields = Dataset(data)

        self.analysis_tables = {}
        self._id_name_mapping = {}

    def _create_empty(self, index=None, columns=None):
        self.additional_fields._create_empty(index, columns)
        self.stats_fields._create_empty(index, columns)
        return self

    def check_hash(self, executor_id: int, space: str) -> bool:
        if space == "additional_fields":
            return executor_id in self.additional_fields.columns
        elif space == "stats_fields":
            return executor_id in self.stats_fields.columns
        elif space == "analysis_tables":
            return executor_id in self.analysis_tables
        else:
            raise ValueError(f"{space} is not a valid space")

    def _create_empty(self, indexes=None, columns=None):
        self.additional_fields._create_empty(indexes, columns)
        self.stats_fields._create_empty(indexes, columns)
        return self

    def set_value(
        self,
        space: str,
        executor_id: int,
        name: str,
        value: Any,
        key: str = None,
        role=None,
    ):
        if space == "additional_fields":
            self.additional_fields.add_column(data=value, name=executor_id, role=role)
        elif space == "analysis_tables":
            self.analysis_tables[name] = value
        elif space == "stats_fields":
            if executor_id not in self.stats_fields.columns:
                self.stats_fields.add_column(
                    data=[None] * len(self.stats_fields),
                    name=executor_id,
                    role=StatisticRole,
                )
            self.stats_fields[executor_id][key] = value
        self._id_name_mapping[executor_id] = name
