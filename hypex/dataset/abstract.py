import copy
import json  # type: ignore
from abc import ABC
from typing import Iterable, Dict, Union, List, Optional, Any

import pandas as pd  # type: ignore

from .backends import PandasDataset
from .roles import (
    ABCRole,
    default_roles,
    DefaultRole,
)
from ..utils import BackendsEnum, RoleColumnError, BackendTypeError


def parse_roles(roles: Dict) -> Dict[Union[str, int], ABCRole]:
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
    """Abstract base class for dataset implementations.

    This class provides the core functionality for managing datasets with roles, data types,
    and different backend storage options.

    Attributes:
        _backend: The backend storage implementation (e.g. PandasDataset)
        _roles (Dict[str, ABCRole]): Mapping of column names to their roles
        _tmp_roles (Dict): Temporary roles used during processing
        default_role (ABCRole): Default role to assign to columns without explicit roles

    Examples:
        Create a dataset with roles:
        >>> roles = {
        ...     'col1': CategoricalRole(),
        ...     'col2': NumericRole()
        ... }
        >>> ds = Dataset(roles=roles, data=pd.DataFrame({
        ...     'col1': ['a', 'b', 'c'],
        ...     'col2': [1, 2, 3]
        ... }))

        Access data and roles:
        >>> ds.data
           col1  col2
        0    a     1
        1    b     2
        2    c     3
        >>> ds.roles
        {'col1': CategoricalRole(), 'col2': NumericRole()}
    """

    @staticmethod
    def _select_backend_from_data(data):
        """Select appropriate backend based on input data.

        Args:
            data: Input data to store in backend

        Returns:
            PandasDataset: Backend dataset implementation

        Examples:
            >>> ds = DatasetBase._select_backend_from_data(pd.DataFrame())
            >>> isinstance(ds, PandasDataset)
            True
        """
        return PandasDataset(data)

    @staticmethod
    def _select_backend_from_str(data, backend):
        """Select backend based on backend enum and data.

        Args:
            data: Input data to store in backend
            backend (BackendsEnum): Enum specifying which backend to use

        Returns:
            PandasDataset: Backend dataset implementation

        Raises:
            TypeError: If backend is not a valid BackendsEnum value

        Examples:
            >>> ds = DatasetBase._select_backend_from_str(
            ...     pd.DataFrame(),
            ...     BackendsEnum.pandas
            ... )
            >>> isinstance(ds, PandasDataset)
            True
        """
        if backend == BackendsEnum.pandas:
            return PandasDataset(data)
        if backend is None:
            return PandasDataset(data)
        raise TypeError("Backend must be an instance of BackendsEnum")

    def _set_all_roles(self, roles):
        """Ensure all columns have roles assigned.

        Assigns default role to any columns not explicitly given roles.

        Args:
            roles (Dict[str, ABCRole]): Current role assignments

        Returns:
            Dict[str, ABCRole]: Updated roles with defaults assigned

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds._set_all_roles({'col1': NumericRole()})
            {'col1': NumericRole(), 'col2': DefaultRole()}
        """
        keys = list(roles.keys())
        for column in self.columns:
            if column not in keys:
                roles[column] = copy.deepcopy(self.default_role) or DefaultRole()
        return roles

    def _set_empty_types(self, roles):
        """Set data types for roles that don't have them specified.

        Infers appropriate data types from the backend data.

        Args:
            roles (Dict[str, ABCRole]): Role assignments to update types for

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole(data_type=None)})
            >>> ds._set_empty_types(ds.roles)
            >>> ds.roles['col1'].data_type
            'float64'
        """
        for column, role in roles.items():
            if role.data_type is None:
                role.data_type = self._backend.get_column_type(column)
            self._backend = self._backend.update_column_type(column, role.data_type)

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
        """Initialize dataset with data and role mappings.

        Args:
            roles: Mapping between columns and their roles. Can be specified in two formats:
                - Dict[ABCRole, Union[List[str], str]]: Role -> column(s) mapping
                - Dict[str, ABCRole]: Column -> role mapping
            data: Input data as DataFrame or path to data file
            backend: Backend storage type to use
            default_role: Role to assign to columns without explicit roles

        Raises:
            TypeError: If roles are not valid ABCRole instances
            RoleColumnError: If role columns don't match data columns

        Examples:
            Basic initialization:
            >>> roles = {'col1': NumericRole()}
            >>> data = pd.DataFrame({'col1': [1,2,3]})
            >>> ds = DatasetBase(roles=roles, data=data)

            With default role:
            >>> ds = DatasetBase(
            ...     roles=roles,
            ...     data=data,
            ...     default_role=CategoricalRole()
            ... )

            With specific backend:
            >>> ds = DatasetBase(
            ...     roles=roles,
            ...     data=data,
            ...     backend=BackendsEnum.pandas
            ... )
        """
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
        self._roles: Dict[str, ABCRole] = roles
        self._tmp_roles: Union[
            Dict[ABCRole, Union[List[str], str]], Dict[Union[List[str], str], ABCRole]
        ] = {}

    def __repr__(self):
        return self.data.__repr__()

    def _repr_html_(self):
        return self.data._repr_html_()

    def __len__(self):
        return self._backend.__len__()

    def search_columns(
        self,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role=False,
        search_types: Optional[List] = None,
    ) -> List[str]:
        """Search for columns matching specified roles and types.

        Args:
            roles: Role or roles to search for
            tmp_role: Whether to search in temporary roles
            search_types: Optional list of data types to filter by

        Returns:
            List[str]: Column names matching the search criteria

        Examples:
            >>> ds = Dataset(roles={
            ...     'col1': NumericRole(),
            ...     'col2': CategoricalRole()
            ... })
            >>> ds.search_columns(NumericRole())
            ['col1']
            >>> ds.search_columns([NumericRole(), CategoricalRole()])
            ['col1', 'col2']
        """
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

    def replace_roles(
        self,
        new_roles_map: Dict[Union[ABCRole, str], ABCRole],
        tmp_role: bool = False,
        auto_roles_types: bool = False,
    ):
        """Replace existing roles with new role mappings.

        Args:
            new_roles_map: Mapping of columns or roles to new roles
            tmp_role: Whether to update temporary roles instead of permanent ones
            auto_roles_types: Whether to automatically infer data types for new roles

        Returns:
            self: Returns self for method chaining

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds.replace_roles({'col1': CategoricalRole()})
            >>> ds.roles['col1']
            CategoricalRole()

            Replace by role type:
            >>> ds.replace_roles({NumericRole(): CategoricalRole()})
        """
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
        """Get the index of the dataset.

        Returns:
            Index: The index of the underlying data

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame(index=['a','b']))
            >>> ds.index
            Index(['a', 'b'], dtype='object')
        """
        return self._backend.index

    @property
    def data(self):
        """Get the underlying data.

        Returns:
            DataFrame: The dataset's data

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame({'col1': [1,2]}))
            >>> ds.data
               col1
            0     1
            1     2
        """
        return self._backend.data

    @property
    def roles(self):
        """Get the role mappings.

        Returns:
            Dict[str, ABCRole]: Mapping of columns to their roles

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds.roles
            {'col1': NumericRole()}
        """
        return self._roles

    @roles.setter
    def roles(self, value):
        self._set_roles(new_roles_map=value, temp_role=False)

    @data.setter
    def data(self, value):
        self._backend.data = value

    @property
    def columns(self):
        """Get dataset column names.

        Returns:
            List[str]: List of column names

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame({'col1': [], 'col2': []}))
            >>> ds.columns
            ['col1', 'col2']
        """
        return self._backend.columns

    @property
    def shape(self):
        """Get dataset dimensions.

        Returns:
            Tuple[int, int]: (rows, columns)

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame({'col1': [1,2,3]}))
            >>> ds.shape
            (3, 1)
        """
        return self._backend.shape

    @property
    def tmp_roles(self):
        """Get temporary role mappings.

        Returns:
            Dict: Temporary role assignments

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds.tmp_roles = {'col1': CategoricalRole()}
            >>> ds.tmp_roles
            {'col1': CategoricalRole()}
        """
        return self._tmp_roles

    @tmp_roles.setter
    def tmp_roles(self, value):
        self._set_roles(new_roles_map=value, temp_role=True)
        self._set_empty_types(self._tmp_roles)

    def to_dict(self):
        """Convert dataset to dictionary format.

        Returns:
            Dict: Dataset as nested dictionary

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds.to_dict()
            {
                'backend': 'pandas',
                'roles': {
                    'role_names': ['col1'],
                    'columns': [NumericRole()]
                },
                'data': {'col1': []}
            }
        """
        return {
            "backend": self._backend.name,
            "roles": {
                "role_names": list(map(lambda x: x, list(self.roles.keys()))),
                "columns": list(self.roles.values()),
            },
            "data": self._backend.to_dict(),
        }

    def to_records(self):
        """Convert dataset to records format.

        Returns:
            List[Dict]: Data as list of record dictionaries

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame({'col1': [1,2]}))
            >>> ds.to_records()
            [{'col1': 1}, {'col1': 2}]
        """
        return self._backend.to_records()

    def to_json(self, filename: Optional[str] = None):
        """Convert dataset to JSON format.

        Args:
            filename: Optional file to write JSON to

        Returns:
            Optional[str]: JSON string if no filename provided

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds.to_json()  # Returns JSON string
            >>> ds.to_json('data.json')  # Writes to file
        """
        if not filename:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    @property
    def backend(self):
        """Get the backend implementation.

        Returns:
            PandasDataset: The backend dataset implementation

        Examples:
            >>> ds = Dataset(roles={})
            >>> isinstance(ds.backend, PandasDataset)
            True
        """
        return self._backend

    def get_values(
        self,
        row: Optional[str] = None,
        column: Optional[str] = None,
    ) -> Any:
        """Get values by row/column labels.

        Args:
            row: Row label
            column: Column label

        Returns:
            Any: Selected values

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame({
            ...     'col1': [1,2],
            ...     'col2': [3,4]
            ... }, index=['a','b']))
            >>> ds.get_values(row='a', column='col1')
            1
            >>> ds.get_values(column='col1')
            [1, 2]
        """
        return self._backend.get_values(row=row, column=column)

    def iget_values(
        self,
        row: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Any:
        """Get values by integer position.

        Args:
            row: Row index
            column: Column index

        Returns:
            Any: Selected values

        Examples:
            >>> ds = Dataset(roles={}, data=pd.DataFrame({
            ...     'col1': [1,2],
            ...     'col2': [3,4]
            ... }))
            >>> ds.iget_values(row=0, column=0)
            1
            >>> ds.iget_values(column=0)
            [1, 2]
        """
        return self._backend.iget_values(row=row, column=column)

    def _set_roles(
        self,
        new_roles_map: Union[
            Dict[ABCRole, Union[List[str], str]], Dict[Union[List[str], str], ABCRole]
        ],
        temp_role: bool = False,
    ):
        """Internal method to update role mappings.

        Args:
            new_roles_map: New role assignments
            temp_role: Whether to update temporary roles

        Returns:
            self: Returns self for method chaining

        Examples:
            >>> ds = Dataset(roles={'col1': NumericRole()})
            >>> ds._set_roles({'col1': CategoricalRole()})
            >>> ds.roles['col1']
            CategoricalRole()
        """
        if not new_roles_map:
            if temp_role:
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
