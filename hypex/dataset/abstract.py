import copy
import json  # type: ignore
from abc import ABC
from typing import Iterable, Dict, Union, List, Optional, Any

import pandas as pd  # type: ignore

from hypex.dataset.backends import PandasDataset
from hypex.dataset.roles import (
    ABCRole,
    default_roles,
    DefaultRole,
)
from hypex.utils import BackendsEnum, RoleColumnError


def parse_roles(roles: Dict) -> Dict[Union[str, int], ABCRole]:
    new_roles = {}
    roles = roles or {}
    for role in roles:
        r = default_roles.get(role, role)
        if isinstance(roles[role], list):
            for i in roles[role]:
                new_roles[i] = copy.deepcopy(r)
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
                roles[column] = copy.deepcopy(self.default_role) or DefaultRole()
        return roles

    def _set_empty_types(self, roles):
        types_map = {
            "int": int,
            "float": float,
            "object": str,
            "bool": bool,
            "category": str,
        }
        reversed_map = {int: "int", float: "float", str: "category", bool: "bool"}
        for column, role in roles.items():
            if role.data_type is None:
                d_type = self.backend.get_column_type(column)

                role.data_type = [v for k, v in types_map.items() if k in d_type][0]
            self._backend = self.backend.update_column_type(
                column, reversed_map[role.data_type]
            )

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
        self._backend = (
            self._select_backend_from_str(data, backend)
            if backend
            else self._select_backend_from_data(data)
        )
        self.default_role = default_role
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
        self._roles: Dict[str, ABCRole] = roles
        self._tmp_roles: Union[
            Dict[ABCRole, Union[List[str], str]], Dict[Union[List[str], str], ABCRole]
        ] = {}

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self._backend.__len__()

    def search_columns(
        self,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role=False,
        search_types: Optional[List] = None,
    ) -> List[str]:
        roles = [roles] if not isinstance(roles, Iterable) else roles
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

    def replace_roles(
        self,
        new_roles_map: Dict[Union[ABCRole, str], ABCRole],
        tmp_role: bool = False,
        auto_roles_types: bool = False,
    ):
        new_roles_map = parse_roles(
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

    @property
    def roles(self):
        return self._roles

    @roles.setter
    def roles(self, value):
        self._set_roles(new_roles_map=value, temp_role=False)

    @data.setter
    def data(self, value):
        self._backend.data = value

    @property
    def columns(self):
        return self._backend.columns

    @property
    def tmp_roles(self):
        return self._tmp_roles

    @tmp_roles.setter
    def tmp_roles(self, value):
        self._set_roles(new_roles_map=value, temp_role=True)
        self._set_empty_types(self._tmp_roles)

    def to_dict(self):
        return {
            "backend": self._backend.name,
            "roles": {
                "role_names": list(map(lambda x: x, list(self.roles.keys()))),
                "columns": list(self.roles.values()),
            },
            "data": self._backend.to_dict(),
        }

    def to_records(self):
        return self._backend.to_records()

    def to_json(self, filename: Optional[str] = None):
        if not filename:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    @property
    def backend(self):
        return self._backend

    def get_values(
        self,
        row: Optional[str] = None,
        column: Optional[str] = None,
    ) -> Any:
        return self._backend.get_values(row=row, column=column)

    def _set_roles(
        self,
        new_roles_map: Union[
            Dict[ABCRole, Union[List[str], str]], Dict[Union[List[str], str], ABCRole]
        ],
        temp_role: bool = False,
    ):
        if not new_roles_map:
            return self.roles

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
                new_roles[columns] = role

        if temp_role:
            self._tmp_roles = new_roles
        else:
            self._roles = new_roles

        return self
