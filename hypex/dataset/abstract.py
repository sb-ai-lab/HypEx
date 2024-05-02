import json
import warnings
from abc import ABC
from typing import Iterable, Dict, Union, List, Optional, Any

import pandas as pd

from hypex.dataset.backends import PandasDataset
from hypex.dataset.roles import (
    InfoRole,
    ABCRole,
    default_roles,
    FeatureRole,
)
from hypex.utils import BackendsEnum, RoleColumnError


# TODO 1: реализовать каскад абстракций (Хранение+Навигация -> Расчеты)
# TODO 2: реализовать каскад абстракций для бэкенда
# TODO 3: магические методы операторов
# TODO 4: математичекие функции (abs, log, mean, median, mode, variance, std, min, max, _shuffle, unique, corr, transpose)
# TODO 5: Enum для статистических тестов + реализация выбора функции из словаря
# TODO 6: для фильтров (n_na, n_unique, value_counts, quantile, select_dtype, join, drop)
# TODO 7: для предобработки (fillna, dropna, sort)


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


class DatasetBase(ABC):
    @staticmethod
    def _select_backend_from_data(data):
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        if backend == BackendsEnum.pandas:
            return PandasDataset(data)
        return PandasDataset(data)

    def _set_all_roles(self, roles):
        keys = list(roles.keys())
        for column in self.columns:
            if column not in keys:
                roles[column] = FeatureRole()
        return roles

    def _set_empty_types(self, roles):
        types_map = {"int": int, "float": float, "object": str, "bool": bool}
        reversed_map = {int: "int", float: "float", str: "category", bool: "bool"}
        for column, role in roles.items():
            if role.data_type is None:
                d_type = self._backend._get_column_type(column)

                role.data_type = [v for k, v in types_map.items() if k in d_type][0]
            self._backend = self._backend._update_column_type(
                column, reversed_map[role.data_type]
            )

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
        if data is not None:
            roles = self._set_all_roles(roles)
            self._set_empty_types(roles)
        self.roles: Dict[Union[str, int], ABCRole] = roles

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self._backend.__len__()

    def __setitem__(self, key: str, value: Any):
        if key not in self.columns and isinstance(key, str):
            self.add_column(value, {key: InfoRole()})
            warnings.warn("Column must be added by add_column", category=SyntaxWarning)
        self.data[key] = value

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
