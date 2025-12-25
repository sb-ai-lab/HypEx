from __future__ import annotations

import copy
import json  # type: ignore
from abc import ABC
from typing import Any, Iterable

import pandas as pd  # type: ignore

from ..utils import BackendsEnum, RoleColumnError
from .backends import PandasDataset
from .roles import ABCRole, DefaultRole, default_roles


def parse_roles(roles: dict) -> dict[str | int] | ABCRole:
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


class DatasetBase(ABC):
    @staticmethod
    def _select_backend_from_data(data):
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        if backend == BackendsEnum.pandas:
            return PandasDataset(data)
        if backend is None:
            return PandasDataset(data)
        raise TypeError("Backend must be an instance of BackendsEnum")

    def _set_all_roles(self, roles):
        keys = list(roles.keys())
        for column in self.columns:
            if column not in keys:
                roles[column] = copy.deepcopy(self.default_role) or DefaultRole()
        return roles

    def _set_empty_types(self, roles):
        for column, role in roles.items():
            if role.data_type is None:
                role.data_type = self._backend.get_column_type(column)
            self._backend = self._backend.update_column_type(column, role.data_type)

    def __init__(
        self,
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole],
        data: pd.DataFrame | str | None = None,
        backend: BackendsEnum | None = None,
        default_role: ABCRole | None = None,
    ):
        self._backend = (
            self._select_backend_from_str(data, backend)
            if backend
            else self._select_backend_from_data(data)
        )
        self.default_role = default_role
        roles = (
            parse_roles(roles)
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
        return self.data.__repr__()

    def _repr_html_(self):
        return self.data._repr_html_()

    def __len__(self):
        return self._backend.__len__()

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
    def shape(self):
        return self._backend.shape

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
