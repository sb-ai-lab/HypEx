from __future__ import annotations

import warnings
import copy
from copy import deepcopy
import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Iterable, Callable, Hashable, Literal, Optional, Sequence
from collections.abc import Iterable as IterableABC

import pandas as pd  # type: ignore
from numpy import ndarray
import pyspark.sql as spark
import pyspark.pandas as ps

from ..utils import (
    BackendsEnum,
    BackendTypeError,
    ConcatBackendError,
    ConcatDataError,
    DataTypeError,
    RoleColumnError,
    NAME_BORDER_SYMBOL,
    ScalarType,
    SourceDataTypes,
    GenericManager
)
from ..utils.adapter import Adapter
from .groupby_dataset import GroupedDataset
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
    DISPLAY_ROWS = 5
    DISPLAY_COLS = 10
    
    @dataclass
    class Locker:
        call_class: Any
        backend: Any
        roles: dict[str, Any]

        def __getitem__(self, item):
            t_data = self.backend.loc(item)
            return self.call_class(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            column_name = item[1]
            if column_name not in self.roles:
                raise KeyError(f"Role for column {column_name} not found")
                
            column_data_type = self.roles[column_name].data_type
            
            is_valid_type = (
                column_data_type is None
                or isinstance(value, column_data_type)
                or (isinstance(value, IterableABC) and all(isinstance(v, column_data_type) for v in value))
            )

            if is_valid_type:
                if column_name not in self.backend.data.columns:
                    raise KeyError("Column must be added by using add_column method.")
                self.backend.data.loc[item] = value
            else:
                raise TypeError("Value type does not match the expected data type.")

    @dataclass
    class ILocker:
        call_class: Any
        backend: Any
        roles: dict[str, Any]

        def __getitem__(self, item):
            t_data = self.backend.iloc(item)
            return self.call_class(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

        def __setitem__(self, item, value):
            column_index = item[1]
            if column_index >= len(self.backend.data.columns):
                raise IndexError("Column index out of range")

            column_name = self.backend.data.columns[column_index]
            column_data_type = self.roles[column_name].data_type
            
            is_valid_type = (
                column_data_type is None
                or isinstance(value, column_data_type)
                or (isinstance(value, IterableABC) and all(isinstance(v, column_data_type) for v in value))
            )

            if is_valid_type:
                self.backend.data.iloc[item] = value
            else:
                raise TypeError("Value type does not match the expected data type.")
    
    @staticmethod
    def _select_backend_from_data(data: Any,
                                  session: spark.SparkSession = None,) -> PandasDataset | SparkDataset:
        if isinstance(data, (PandasDataset, SparkDataset)):
            return data
        elif isinstance(data, pd.DataFrame):
            return PandasDataset(data)
        elif isinstance(data, (spark.DataFrame, ps.DataFrame)):
            return SparkDataset(data, session)

        raise TypeError("data must be an instance of either"
                        "pandas.DataFrame, spark.DataFrame or PandasDataset, SparkDataset")

    def _select_non_encoding_columns(
        self,
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole] | None = None,
    ) -> list[str] | None:
        columns = None
        if isinstance(roles, dict):
            if any(isinstance(role, ABCRole) for role in roles.keys()):
                roles = self._parse_roles(roles)
            columns = []
            for column, role in roles.items():
                if role.data_type == str:
                    columns.append(column)
        return columns

    @staticmethod
    def _select_backend_from_str(
        data: Any,
        backend: BackendsEnum | None,
        data_compression: Literal["downcasting", "encoding", "auto", "disable"],
        session: spark.SparkSession | None = None,
        non_compresion_cols: list[str] | None = None,
    ) -> PandasDataset | SparkDataset:
        if backend == BackendsEnum.pandas:
            return PandasDataset(data, data_compression, non_compresion_cols)
        elif backend == BackendsEnum.spark:
            return SparkDataset(data, session)
        elif backend is None:
            if session is not None:
                return SparkDataset(data, session)
            return PandasDataset(data, data_compression, non_compresion_cols)
        raise TypeError("Backend must be an instance of BackendsEnum")

    def _set_all_roles(self,
                       roles: dict[str, ABCRole]) -> dict[str, ABCRole]:
        keys = list(roles.keys())
        for column in self.columns:
            if column not in keys:
                roles[column] = copy.deepcopy(self.default_role) or DefaultRole()
        return roles

    def _set_empty_types(self,
                         roles: dict[str, ABCRole]) -> None:
        colunms_dtypes = self._backend_data.get_column_type(self._backend_data.columns)
        new_types = {}
        for column, role in roles.items():
            if role.data_type is None:
                role.data_type = colunms_dtypes[column]
            elif role.data_type != colunms_dtypes[column]:
                new_types[column] = role.data_type
        self._backend_data = self._backend_data.update_column_type(new_types)

    def __init__(self,
                 roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole] | None = None,
                 data: spark.DataFrame | pd.DataFrame | str | DatasetBase | None = None,
                 backend: BackendsEnum | None = None,
                 default_role: ABCRole | None = None,
                 session: Optional[spark.SparkSession] = None,
                 data_compression: Literal["downcasting", "encoding", "auto", "disable"] = "auto"):
        if backend is not None:
            non_compresion_cols = self._select_non_encoding_columns(roles)
            self._backend_data = self._select_backend_from_str(data, backend, data_compression,session, non_compresion_cols)
        elif data is not None:
            if isinstance(data, DatasetBase):
                self._backend_data = copy.deepcopy(data._backend)
            elif isinstance(data, (PandasDataset, SparkDataset)):
                self._backend_data = data
            elif any(isinstance(data, source_data_type)
                     for source_data_type in SourceDataTypes.__args__):
                self._backend_data = self._select_backend_from_data(data, session)
            else:
                if session is not None:
                    self._backend_data = SparkDataset(data, session)
                else:
                    self._backend_data = PandasDataset(data)
        else:
            if session is not None:
                self._backend_data = SparkDataset(data, session)
            else:
                self._backend_data = PandasDataset(data)

        self.default_role = default_role
        if roles is None and data.hasattr("roles") and data.roles is not None:
            roles = data.roles
        elif roles is None:
            roles = {}
        else:
            if any(isinstance(role, ABCRole) for role in roles.keys()):
                roles = self._parse_roles(roles)

            if any(not isinstance(role, ABCRole) for role in roles.values()):
                raise TypeError("Roles must be instances of ABCRole type")

            if data is not None and hasattr(self._backend_data, 'columns'):
                invalid_columns = [i for i in roles.keys() if i not in self._backend_data.columns]
                if invalid_columns:
                    raise RoleColumnError(invalid_columns, self._backend_data.columns)

        if data is not None and hasattr(self._backend_data, 'columns'):
            roles = self._set_all_roles(roles)
            self._set_empty_types(roles)

        self._roles: dict[str, ABCRole] = roles
        self._tmp_roles: dict[str, ABCRole] = {}
        
        self.loc = self.Locker(call_class=self.__class__, backend=self._backend_data, roles=self.roles)
        self.iloc = self.ILocker(call_class=self.__class__, backend=self._backend_data, roles=self.roles)
        
    def persist(self, 
                storage_level: Literal["MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY", 
                                      "MEMORY_ONLY_SER", "MEMORY_AND_DISK_SER", 
                                      "OFF_HEAP"] = "MEMORY_AND_DISK",
                action: Literal["count", "head", "none"] = "count"):
        if self.backend_type == BackendsEnum.spark:
            self._backend_data.persist(storage_level=storage_level, action=action)
        
        return self

    def unpersist(self, blocking: bool = False):
        if self.backend_type == BackendsEnum.spark:
            self._backend_data.unpersist(blocking=blocking)
        
        return self
    
    @property
    def is_persisted(self) -> bool:
        if self.backend_type == BackendsEnum.spark:
            return self._backend_data.is_persisted
        return False

    def get_storage_level(self) -> str | None:

        if self.backend_type == BackendsEnum.spark:
            return self._backend_data.get_storage_level()
        return None

    def get_cache_info(self) -> dict[str, Any]:
        return {
            "is_persisted": self.is_persisted(),
            "storage_level": self.get_storage_level(),
            "backend_type": self.backend_type,
            "in_memory": True if self.backend_type == BackendsEnum.pandas else self.is_persisted(),
        }

    def __repr__(self):
        n_cols = len(self.columns)
        n_rows = self._backend_data.shape[0]
        df = self._build_repr(n_cols, n_rows)
        return f"{df.to_string()}\n\n{n_rows} rows × {n_cols} columns"
        # return self.backend_data.__repr__

    def _repr_html_(self):
        n_cols = len(self.columns)
        n_rows = self._backend_data.shape[0]
        df = self._build_repr(n_cols, n_rows)
        html_table = df.to_html()
        html_info = f'''
        <div style="font-size: 12px; color: #bdbdbd; margin-top: 5px; font-family: monospace;">
            {n_rows} rows × {n_cols} columns
        </div>
        '''
        return html_table + html_info

    def __len__(self) -> int:
        return self._backend_data.__len__()

    def __getitem__(self,
                    item: str | int | Iterable[str | int] | slice | DatasetBase) -> DatasetBase:
        if isinstance(item, DatasetBase):
            item = item.data
        elif isinstance(item, slice):
            result = self._backend_data.__getitem__(item)
            return self.__class__(roles=self.roles, data=result)
            
        if isinstance(item, (pd.DataFrame, ps.DataFrame, pd.Series, ps.Series)):
            result_data = self._backend_data.__getitem__(item)
            res_cols = list(result_data.columns) if hasattr(result_data, 'columns') else self.columns
            roles = {col: self.roles.get(col, InfoRole()) for col in res_cols}
            result = self.__class__(data=result_data, roles=roles)
            result.tmp_roles = {
                k: v for k, v in self.tmp_roles.items() if k in result.columns
            }
            return result

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
        result = self.__class__(data=self._backend_data.__getitem__(item), roles=roles)
        result.tmp_roles = {
            key: value for key, value in self.tmp_roles.items() if key in items
        }
        return result
    
    def reset_index(self,
                    drop: bool = False,
                    **kwargs) -> DatasetBase:
        kwargs['inplace'] = False
        
        new_data = self._backend_data.reset_index(drop=drop, **kwargs)
        
        new_roles = deepcopy(self.roles)
        
        if not drop:
            for col in new_data.columns:
                if col not in new_roles:
                    new_roles[col] = deepcopy(self.default_role) or InfoRole()
        
        return self.__class__(roles=new_roles, data=new_data)

    def __setitem__(self,
                    key: str,
                    value: Any) -> None:
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
                )
                or isinstance(value, column_data_type)
            ):
                self.data[key] = value
            else:
                raise TypeError("Value type does not match the expected data type.")

    def _build_repr(self, n_cols, n_rows) -> pd.DataFrame:
        display_limit = n_rows if n_rows <= self.DISPLAY_ROWS * 2 else self.DISPLAY_ROWS
        head = self._backend_data._display_head_tail(
            rows_display_limit=display_limit,
            cols_display_limit=self.DISPLAY_COLS,
            n_cols=n_cols,
            n_rows=n_rows)

        if n_rows > self.DISPLAY_ROWS * 2:
            _tmp_tail = self._backend_data._display_head_tail(
                rows_display_limit=self.DISPLAY_ROWS,
                cols_display_limit=self.DISPLAY_COLS,
                n_cols=n_cols,
                n_rows=n_rows,
                tail=True)

            tail = pd.concat([pd.DataFrame([["..."] * len(head.columns)],
                                           index=["..."],
                                           columns=head.columns),
                            _tmp_tail], axis=0)
            return pd.concat([head, tail], axis=0)
        else:
            return head

    @classmethod
    def create_empty(cls,
                     roles: dict[str, ABCRole] | None = None,
                     index=None,
                     session=None,
                     backend=BackendsEnum.pandas) -> DatasetBase:
        if roles is None:
            roles = {}
        index = [] if index is None else index
        columns = list(roles.keys())
        ds = cls(roles=roles, backend=backend, session=session)
        ds._backend_data = ds._backend_data.create_empty(index, columns)
        ds.data = ds._backend_data.data
        return ds

    @staticmethod
    def _parse_roles(roles: dict[Any, Any]) -> dict[str, ABCRole]:
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
    
    

    def get(self,
            key: Any,
            default: Any = None) -> DatasetBase:
        return self.__class__(
            data=self._backend_data.get(key, default), roles=deepcopy(self.roles)
        )

    def take(self,
             indices: int | list[int],
             axis: Literal["index", "columns", "rows"] | int = 0) -> DatasetBase:
        new_data = self._backend_data.take(indices=indices, axis=axis)
        new_roles = (
            {k: deepcopy(v) for k, v in self.roles.items() if k in new_data.columns}
            if axis == 1
            else deepcopy(self.roles)
        )
        return self.__class__(data=new_data, roles=new_roles)
    
    def __binary_magic_operator(self, other: Any, func_name: str) -> Any:
        if not any(isinstance(other, t) for t in [self.__class__, str, int, float, bool, Sequence]):
            raise DataTypeError(type(other))

        self_raw = self.backend_data.data if hasattr(self.backend_data, 'data') else self.backend_data

        if isinstance(other, self.__class__):
            if type(other._backend_data) is not type(self._backend_data):
                raise BackendTypeError(type(other._backend_data), type(self._backend_data))
            other_raw = other.backend_data.data if hasattr(other.backend_data, 'data') else other.backend_data
            
            if hasattr(other_raw, 'columns') and hasattr(self_raw, 'columns'):
                if len(other_raw.columns) == len(self_raw.columns) and list(other_raw.columns) != list(self_raw.columns):
                    rename_map = {other_raw.columns[i]: self_raw.columns[i] for i in range(len(self_raw.columns))}
                    other_raw = other_raw.rename(columns=rename_map)
                if hasattr(self_raw, 'columns') and not hasattr(other_raw, 'columns'):
                    other_raw = other_raw.to_frame()
                    if other_raw.columns[0] != self_raw.columns[0]:
                        other_raw.columns = self_raw.columns
        else:
            other_raw = other

        func = getattr(self_raw, func_name)
        result_raw = func(other_raw)

        if hasattr(result_raw, 'to_frame') and not hasattr(result_raw, 'columns'):
            col_name = result_raw.name if result_raw.name is not None else (self_raw.columns[0] if hasattr(self_raw, 'columns') else 'result')
            result_raw = result_raw.to_frame(name=col_name)

        actual_columns = list(result_raw.columns) if hasattr(result_raw, 'columns') else []
        new_roles = {}
        for col in actual_columns:
            new_roles[col] = deepcopy(self.roles.get(col, self.default_role or DefaultRole()))
            new_roles[col].data_type = None

        return self.__class__(roles=new_roles, data=result_raw, session=self.session)

    # comparison operators:
    def __eq__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__eq__")

    def __ne__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__ne__")

    def __le__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__le__")

    def __lt__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__lt__")

    def __ge__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__ge__")

    def __gt__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__gt__")

    # unary operators:
    def __pos__(self) -> DatasetBase:
        return self.__class__(roles=self.roles, data=(+self._backend_data), session=self.session)

    def __neg__(self) -> DatasetBase:
        return self.__class__(roles=self.roles, data=(-self._backend_data), session=self.session)

    def __abs__(self) -> DatasetBase:
        return self.__class__(roles=self.roles, data=abs(self._backend_data), session=self.session)

    def __invert__(self) -> DatasetBase:
        return self.__class__(roles=self.roles, data=(~self._backend_data), session=self.session)

    def __round__(self, ndigits: int = 0) -> DatasetBase:
        return self.__class__(roles=self.roles, data=round(self._backend_data, ndigits), session=self.session)

    def __bool__(self) -> DatasetBase:
        return not self._backend_data.is_empty()

    # Binary math operators:
    def __add__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__add__")

    def __sub__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__sub__")

    def __mul__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__mul__")

    def __floordiv__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__floordiv__")

    def __div__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__div__")

    def __truediv__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__truediv__")

    def __mod__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__mod__")

    def __pow__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__pow__")

    def __and__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__and__")

    def __or__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__or__")

    # Right math operators:
    def __radd__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__radd__")

    def __rsub__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rsub__")

    def __rmul__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rmul__")

    def __rfloordiv__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rfloordiv__")

    def __rdiv__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rdiv__")

    def __rtruediv__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rtruediv__")

    def __rmod__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rmod__")

    def __rpow__(self, other: Any) -> DatasetBase:
        return self.__binary_magic_operator(other=other, func_name="__rpow__")
    
    def __deepcopy__(self, memo):
        """deepcopy dataset"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k.startswith('_abc_'):
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def search_columns(self,
                       roles: ABCRole | Iterable[ABCRole],
                       tmp_role: bool =False,
                       search_types: list[type] | None = None) -> list[str]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        # roles_for_search = self._tmp_roles if tmp_role else self.roles
        roles_for_search = {**self.roles, **self._tmp_roles} if tmp_role else self.roles
        return [
            str(column)
            for column, role in roles_for_search.items()
            if any(
                isinstance(r, role.__class__)
                and (not search_types or role.data_type in search_types)
                for r in roles
            )
        ]

    def search_columns_by_type(self,
                               search_types: list[type] | type) -> list[str]:
        search_types = (
            search_types if isinstance(search_types, Iterable) else [search_types]
        )
        return [
            str(column)
            for column, role in self.roles.items()
            if any(role.data_type == t for t in search_types)
        ]

    def replace_roles(self,
                      new_roles_map: dict[ABCRole, list[str] | str] | dict[str, ABCRole] | ABCRole,
                      tmp_role: bool = False,
                      auto_roles_types: bool = False):
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
    def index(self) -> Any:
        return self._backend_data.index
    
    @index.setter
    def index(self, value):
        self._backend_data.index = value

    @property
    def data(self) -> pd.DataFrame | spark.DataFrame:
        return self._backend_data.data

    @data.setter
    def data(self, value: pd.DataFrame | spark.DataFrame) -> None:
        self._backend_data.data = value

    @property
    def columns(self) -> list[str]:
        return self._backend_data.columns

    @property
    def roles(self) -> dict[str, ABCRole]:
        return self._roles

    @roles.setter
    def roles(self, value: dict[str, ABCRole]) -> None:
        self._set_roles(new_roles_map=value, temp_role=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self._backend_data.shape

    @property
    def tmp_roles(self) -> dict[str, ABCRole]:
        return self._tmp_roles

    @property
    def session(self) -> spark.SparkSession | None:
        return self._backend_data.session

    @property
    def labels_dict(self):
        return self._backend_data.labels_dict

    @property
    def backend_type(self):
        if isinstance(self._backend_data, PandasDataset):
            return BackendsEnum.pandas
        elif isinstance(self._backend_data, SparkDataset):
            return BackendsEnum.spark
        else:
            raise ValueError("Unknown backend type")

    @tmp_roles.setter
    def tmp_roles(self, value: dict[str, ABCRole]) -> None:
        self._set_roles(new_roles_map=value, temp_role=True)
        self._set_empty_types(self._tmp_roles)

    def _convert_data_after_agg(self,
                                result: Any) -> DatasetBase | ScalarType | None:
        if result is None:
            return None

        # if isinstance(result, ScalarType):
        if GenericManager.check_type(result, ScalarType):
            return result

        if not hasattr(result, 'columns'):
            return result

        role: ABCRole = StatisticRole()
        return self.__class__(
            data=result, roles={column: role for column in result.columns}
        )

    def add_column(self,
                   data: Any,
                   role: dict[str, ABCRole] | None = None,
                   index: Iterable[Hashable] | None = None):
        if role is None:
            if not isinstance(data, self.__class__):
                raise ValueError("If role is None, data must be a Dataset")
            if any([col in self.columns for col in data.columns]):
                raise ValueError("Columns with the same name already exist")
            self.roles.update(data.roles)
            self._backend_data.add_column(
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
            self._backend_data.add_column(data, list(role.keys()), index)
        return self

    def _check_other_dataset(self, other):
        if not isinstance(other, self.__class__):
            raise ConcatDataError(type(other))
        if type(other._backend_data) is not type(self._backend_data):
            raise ConcatBackendError(type(other._backend_data), type(self._backend_data))

    def astype(self,
               dtype: dict[str, type],
               errors: Literal["raise", "ignore"] = "raise") -> DatasetBase:
        for col, _ in dtype.items():
            if (errors == "raise") and (col not in self.columns):
                raise KeyError(f"Column '{col}' does not exist in the Dataset.")

        new_data = self._backend_data.astype(dtype, errors)

        new_roles = deepcopy(self.roles)

        for col, target_type in dtype.items():
            if col in new_roles:
                new_roles[col].data_type = target_type

        return self.__class__(roles=new_roles, data=new_data)

    def append(self,
               other: DatasetBase | Iterable[DatasetBase],
               reset_index=False,
               axis: int = 0) -> DatasetBase:
        other = Adapter.to_list(other)

        new_roles = deepcopy(self.roles)
        for o in other:
            self._check_other_dataset(o)
            new_roles.update(o.roles)

        return self.__class__(roles=new_roles, 
                              data=self.backend_data.append(other=other, reset_index=reset_index, axis=axis))
    
    def limit(self, num: int | None = None) -> Any:
        return self.__class__(data=self._backend_data.limit(num=num), roles=self.roles)

    def apply(self,
              func: Callable[..., Any],
              role: dict[str, ABCRole],
              axis: int = 0,
              **kwargs) -> DatasetBase:
        if self.is_empty():
            return deepcopy(self)
        tmp_data = self._backend_data.apply(
            func=func, axis=axis, column_name=next(iter(role.keys())), **kwargs
        )
        tmp_roles = deepcopy(role)
        return self.__class__(
            data=tmp_data,
            roles=tmp_roles,
        )

    def map(self,
            func: Callable[..., Any],
            na_action: Any = None,
            **kwargs) -> DatasetBase:
        return self.__class__(
            roles=self.roles,
            data=self._backend_data.map(func=func, na_action=na_action, **kwargs),
        )

    def is_empty(self) -> bool:
        return self._backend_data.is_empty()

    def unique(self) -> dict[str, list[Any]]:
        return self._backend_data.unique()

    def nunique(self, dropna: bool = False) -> dict[str, int]:
        return self._backend_data.nunique(dropna)

    def isin(self, values: Iterable[Any]) -> DatasetBase:
        role: ABCRole = FilterRole()
        return self.__class__(
            roles={column: role for column in self.roles.keys()},
            data=self._backend_data.isin(values),
        )

    def groupby(self,
                by: str | Iterable[str],
                **kwargs) -> GroupedDataset:
        return GroupedDataset(
            backend_groupby=self._backend_data.groupby(by=by, **kwargs),
            dataset_class=self.__class__,
            roles=self.roles,
            tmp_roles=self.tmp_roles,
            group_cols=Adapter.to_list(by),
            backend_data=self._backend_data
        )

    def fillna(self,
               values: ScalarType | dict[str, ScalarType] | None = None,
               method: Literal["bfill", "ffill"] | None = None,
               **kwargs) -> DatasetBase:
        if values is None and method is None:
            raise ValueError("Value or filling method must be provided")
        return self.__class__(
            roles=self.roles,
            data=self.backend_data.fillna(values=values, method=method, **kwargs),
        )

    def mean(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.mean())

    def max(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.max())

    def min(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.min())

    def count(self) -> DatasetBase | ScalarType | None:
        if self.is_empty():
            return self.create_empty({role: InfoRole() for role in self.roles})
        return self._convert_data_after_agg(self._backend_data.count())

    def sum(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.sum())

    def log(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.log())

    def mode(self,
             numeric_only: bool = False,
             dropna: bool = True) -> DatasetBase:
        t_data = self._backend_data.mode(numeric_only=numeric_only, dropna=dropna)
        return self.__class__(
            data=t_data, roles={role: InfoRole() for role in t_data.columns}
        )

    def var(self,
            skipna: bool = True,
            ddof: int = 1,
            numeric_only: bool = False) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(
            self._backend_data.var(skipna=skipna, ddof=ddof, numeric_only=numeric_only)
        )

    def agg(self,
            func: str | list) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.agg(func))

    def std(self,
            skipna: bool = True,
            ddof: int = 1) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.std(skipna=skipna, ddof=ddof))

    def quantile(self,
                 q: float = 0.5) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.quantile(q=q))

    def coefficient_of_variation(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.coefficient_of_variation())

    def corr(self, numeric_only: bool = False) -> DatasetBase:
        t_data = self._backend_data.corr(numeric_only=numeric_only)
        t_roles = {column: self.roles[column] for column in t_data.columns}
        return self.__class__(roles=t_roles, data=t_data)

    def value_counts(self,
                     normalize: bool = False,
                     sort: bool = True,
                     ascending: bool = False,
                     dropna: bool = True) -> DatasetBase:
        t_data = self._backend_data.value_counts(
            normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
        )
        new_columns = t_data.columns if hasattr(t_data, 'columns') else list(t_data.data) 

        t_roles = {
            col_name: role 
            for col_name, role in self.roles.items() 
            if col_name in new_columns
        }

        column_name = "proportion" if normalize else "count"
        if column_name not in new_columns:
            t_data = t_data.rename(columns={0: column_name})
            
        t_roles[column_name] = StatisticRole()
        return self.__class__(roles=t_roles, data=t_data)

    def na_counts(self) -> DatasetBase | ScalarType | None:
        """Count NA values"""
        return self._convert_data_after_agg(self._backend_data.na_counts())

    def isna(self) -> DatasetBase | ScalarType | None:
        return self._convert_data_after_agg(self._backend_data.isna())

    def dropna(self,
               how: Literal["any", "all"] = "any",
               subset: str | Iterable[str] | None = None,
               axis: Literal["index", "rows", "columns"] | int = 0) -> DatasetBase:
        new_data = self._backend_data.dropna(how=how, subset=subset, axis=axis)

        new_roles = (
            self.roles
            if axis == 0
            else {column: self.roles[column] for column in new_data.columns}
        )

        return self.__class__(roles=new_roles, data=new_data)

    def drop(self,
             labels: str | None = None,
             axis: int | None = None,
             columns: str | Iterable[str] | None = None) -> DatasetBase:
        dropped_columns = []
        
        if columns is not None:
            dropped_columns = Adapter.to_list(columns)
        elif labels is not None and axis == 1:
            dropped_columns = Adapter.to_list(labels)
        
        new_data = self._backend_data.drop(labels=labels, axis=axis, columns=columns)
        
        new_roles = {
            column: deepcopy(role) 
            for column, role in self.roles.items() 
            if column not in dropped_columns and column in new_data.columns
        }
        
        new_tmp_roles = {
            column: deepcopy(role)
            for column, role in self._tmp_roles.items()
            if column not in dropped_columns and column in new_data.columns
        }
        
        result = self.__class__(roles=new_roles, data=new_data)
        result._tmp_roles = new_tmp_roles
        return result

    def filter(self, items = None, regex = None, axis = None):
        t_data = self._backend_data.filter(items=items, regex=regex, axis=axis)
        t_roles = {c: self.roles[c] for c in t_data.columns if c in self.roles.keys()}
        
        raw_data = t_data.data if hasattr(t_data, 'data') else t_data
        
        return self.__class__(roles=t_roles, data=raw_data)

    def select(self, columns: str | list[str]):
        columns = Adapter.to_list(columns)
        return self.filter(items=columns, axis=1)

    def iselect(self, columns: int | list[int]):
        columns = Adapter.to_list(columns)
        columns = [self.columns[n] for n in columns]
        return self.filter(items=columns, axis=1)

    def select_dtypes(self, include: Any = None, exclude: Any = None):
        t_data = self._backend_data.select_dtypes(include=include, exclude=exclude)

        t_roles = {k: v for k, v in self.roles.items() if k in t_data.columns}
        return self.__class__(roles=t_roles, data=t_data)

    def merge(self,
              right,
              on: str | None = None,
              left_on: str | None = None,
              right_on: str | None = None,
              left_index: bool = False,
              right_index: bool = False,
              suffixes: tuple[str, str] = ("_x", "_y"),
              how: Literal["left", "right", "outer", "inner", "cross"] = "inner") -> DatasetBase:
        if not isinstance(right, self.__class__):
            raise DataTypeError(type(right))
        if type(right._backend_data) is not type(self._backend_data):
            raise BackendTypeError(type(right._backend_data), type(self._backend_data))

        t_data = self._backend_data.merge(
            right=right._backend_data,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            how=how,
        )

        t_roles = deepcopy(self.roles)
        t_roles.update(right.roles)

        for c in t_data.columns:
            if f"{c}".endswith(suffixes[0]) and c[: -len(suffixes[0])] in self.columns:
                t_roles[c] = self.roles[c[: -len(suffixes[0])]]
            if f"{c}".endswith(suffixes[1]) and c[: -len(suffixes[1])] in right.columns:
                t_roles[c] = right.roles[c[: -len(suffixes[1])]]

        new_roles = {c: t_roles[c] for c in t_data.columns}
        return self.__class__(roles=new_roles, data=t_data)

    def dot(self, other: DatasetBase | ndarray) -> DatasetBase:
        return self.__class__(
            roles=deepcopy(other.roles) if isinstance(other, self.__class__) else {},
            data=self.backend_data.dot(other.backend_data if isinstance(other, self.__class__) else other),
        )

    def sample(self,
               frac: float | None = None,
               n: int | None = None,
               random_state: int | None = None) -> DatasetBase:
        return self.__class__(
            self.roles,
            data=self.backend_data.sample(frac=frac, n=n, random_state=random_state),
        )

    def cov(self) -> DatasetBase:
        t_data = self.backend_data.cov()
        return self.__class__(
            {column: DefaultRole() for column in t_data.columns}, data=t_data
        )

    def rename(self, names: dict[str, str]) -> DatasetBase:
        roles = {names.get(column, column): role for column, role in self.roles.items()}
        return self.__class__(roles, data=self.backend_data.rename(names))

    def replace(self,
                to_replace: Any = None,
                value: Any = None,
                regex: bool = False) -> DatasetBase:
        return self.__class__(
            self.roles,
            data=self._backend_data.replace(to_replace=to_replace, value=value, regex=regex),
        )

    def list_to_columns(self, column: str) -> DatasetBase:
        if not pd.api.types.is_list_like(self.backend_data[column][0]):
            return self
        extended_data = self.backend_data.list_to_columns(column)
        extended_roles = {
            c: deepcopy(self.roles[column]) for c in extended_data.columns
        }
        extended_ds = self.__class__(roles=extended_roles, data=extended_data)
        return self.append(extended_ds, axis=1).drop(column, axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self._backend_data.name,
            "roles": {
                "role_names": list(map(lambda x: x, list(self.roles.keys()))),
                "columns": list(self.roles.values()),
            },
            "data": self._backend_data.to_dict(),
        }

    def to_numpy(self) -> ndarray:
        return self._backend_data.to_numpy()

    def to_records(self) -> Any:
        return self._backend_data.to_records()

    def to_json(self, filename: str | None = None):
        if not filename:
            return json.dumps(self.to_dict())
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file)

    @property
    def backend_data(self) -> PandasDataset | SparkDataset:
        return self._backend_data

    def get_values(self,
                   row: str | None = None,
                   column: str | None = None) -> Any:
        return self._backend_data.get_values(row=row, column=column)

    def iget_values(self,
                    row: int | None = None,
                    column: int | None = None) -> Any:
        return self._backend_data.iget_values(row=row, column=column)

    def _set_roles(self,
                   new_roles_map: dict[ABCRole, list[str] | str] | dict[list[str] | str] | ABCRole,
                   temp_role: bool = False) -> DatasetBase:
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
