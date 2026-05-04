from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd  # type: ignore
import pyspark.sql as spark
import pyspark.pandas as ps

from ..utils import (
    ID_SPLIT_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    MultiFieldKeyTypes,
    NotFoundInExperimentDataError,
    ScalarType,
)
from ..utils.adapter import Adapter
from ..utils.errors import InvalidArgumentError
from .abstract import DatasetBase
from .roles import (
    ABCRole,
    AdditionalRole,
    DefaultRole,
)

from typing import Literal

class Dataset(DatasetBase):
    def __init__(
        self,
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole],
        data: pd.DataFrame | spark.DataFrame | str | None = None,
        backend: BackendsEnum | None = None,
        default_role: ABCRole | None = None,
        session: spark.SparkSession | None = None,
        data_compression: Literal["downcasting", "encoding", "auto", "disable"] = "auto"
    ):
        super().__init__(roles, data, backend, default_role, session, data_compression)

    def to_small_dataset(self) -> SmallDataset:
        return SmallDataset(
            roles=self.roles,
            data=self.data,
            default_role=self.default_role,
        )


class SmallDataset(DatasetBase):
    def __init__(
        self,
        roles: dict[ABCRole, list[str] | str] | dict[str, ABCRole],
        data: pd.DataFrame | str | None = None,
        default_role: ABCRole | None = None,
        session: spark.SparkSession | None = None,
    ):
        if isinstance(roles, dict) and data is not None:
            columns = None
            
            if hasattr(data, 'columns'):
                cols = data.columns
                columns = cols.tolist() if hasattr(cols, 'tolist') else list(cols)
            elif hasattr(data, 'name') and data.name is not None:
                columns = [data.name]
            elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict):
                columns = list(data['data'].keys())
            
            if columns:
                new_roles = {}
                for k, v in roles.items():
                    if isinstance(k, int) and 0 <= k < len(columns):
                        new_roles[columns[k]] = v
                    elif not isinstance(k, int):
                        new_roles[k] = v
                roles = new_roles
        if isinstance(data, pd.Series):
            if data.name is None:
                data = data.to_frame(name="value")
            else:
                data = data.to_frame()
        
        super().__init__(roles, data, BackendsEnum.pandas, default_role, session)
        self.loc = self.Locker(call_class=self.__class__, backend=self._backend_data, roles=self.roles)
        self.iloc = self.ILocker(call_class=self.__class__, backend=self._backend_data, roles=self.roles)

    @property
    def index(self):
        return self.backend_data.index

    @index.setter
    def index(self, value):
        self.backend_data.data.index = value

    @staticmethod
    def from_dict(
            data: FromDictTypes,
            roles: ABCRole | dict[str, ABCRole],
    ) -> SmallDataset:
        if not isinstance(roles, dict):
            if isinstance(roles, ABCRole):
                roles = {col : roles for col in data.keys()}
            else:
                raise TypeError(f"Value {roles} is not a dict type.")

        if isinstance(data, dict) and "data" in data:
            payload = data
        elif isinstance(data, dict):
            payload = {"data": data}
        else:
            payload = data
        return SmallDataset(data=payload, roles=roles)

    def sort(
        self,
        by: MultiFieldKeyTypes | None = None,
        ascending: bool = True,
        **kwargs,
    ):
        if by is None:
            return Dataset(
                roles=self.roles,
                data=self.backend.sort_index(ascending=ascending, **kwargs),
            )
        return Dataset(
            roles=self.roles,
            data=self.backend.sort_values(by=by, ascending=ascending, **kwargs),
        )

    def reindex(self, labels, fill_value: Any | None = None) -> Dataset:
        return Dataset(
            self.roles, data=self.backend.reindex(labels, fill_value=fill_value)
        )

    def idxmax(self):
        return self._convert_data_after_agg(self._backend_data.idxmax())

    def transpose(
            self,
            roles: dict[str, ABCRole] | list[str] | None = None,
    ) -> Dataset:
        roles_names: list[str | None] = (
            list(roles.keys()) or [] if isinstance(roles, dict) else roles
        )
        result_data = self.backend_data.transpose(roles_names)
        if roles is None or isinstance(roles, list):
            names = result_data.columns if roles is None else roles
            roles = {column: DefaultRole() for column in names}

        return Dataset(roles=roles, data=result_data)

    def to_dataset(self) -> Dataset:
        return Dataset(
            roles=self.roles,
            data=self.data,
            default_role=self.default_role,
        )


class ExperimentData:
    def __init__(self, data: Dataset | SmallDataset):
        self._data: Dataset | SmallDataset = data
        self.additional_fields: Dataset | SmallDataset = (data
            .create_empty(index=data.index, 
                          backend=data.backend_type,
                          session=data.session))
        self.variables: dict[str, dict[str, int | float]] = {}
        self.groups: dict[str, dict[str, Dataset]] = {}
        self.analysis_tables: dict[str, SmallDataset] = {}
        self.id_name_mapping: dict[str, str] = {}

    @property
    def ds(self) -> Dataset | SmallDataset:
        """
        Get the base dataset.
        """
        return self._data

    @staticmethod
    def create_empty(roles=None, backend=BackendsEnum.pandas, index=None) -> ExperimentData:
        if isinstance(index, Dataset):
            index = index.index
        ds = Dataset.create_empty(backend, roles, index)
        return ExperimentData(ds)

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
        if space == ExperimentDataEnum.additional_fields:
            return executor_id in self.additional_fields.columns
        elif space == ExperimentDataEnum.variables:
            return executor_id in self.variables.keys()
        elif space == ExperimentDataEnum.analysis_tables:
            return executor_id in self.analysis_tables
        else:
            return any(self.check_hash(executor_id, s) for s in ExperimentDataEnum)

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: str | dict[str, str],
        value: Any,
        key: str | None = None,
        role=None,
    ) -> ExperimentData:
        # Handle additional fields
        if space == ExperimentDataEnum.additional_fields:
            if not isinstance(value, Dataset):
                self.additional_fields = self.additional_fields.add_column(
                    data=value, role={executor_id: role}
                )
            elif len(value.columns) == 1:
                role = role[0] if isinstance(role, list) else role
                role = next(iter(role.values())) if isinstance(role, dict) else role
                executor_id = (
                    executor_id[0] if isinstance(executor_id, list) else executor_id
                )
                executor_id = (
                    next(iter(executor_id.keys()))
                    if isinstance(executor_id, dict)
                    else executor_id
                )
                # self.additional_fields.unpersist()
                self.additional_fields = self.additional_fields.add_column(
                    data=value, role={executor_id: role}
                )
                # self.additional_fields.persist()
            else:
                rename_dict = (
                    {value.columns[0]: executor_id}
                    if isinstance(executor_id, str)
                    else executor_id
                )
                value = value.rename(names=rename_dict)
                self.additional_fields = self.additional_fields.merge(
                    right=value, left_index=True, right_index=True
                )

        # Handle analysis tables
        elif space == ExperimentDataEnum.analysis_tables:
            # Преобразуем Dataset в SmallDataset
            if isinstance(value, Dataset):
                value = value.to_small_dataset()
            elif isinstance(value, Dataset):
                value = SmallDataset.from_dict(value.to_dict(), roles=role)
            elif not isinstance(value, SmallDataset):
                # Если значение не Dataset/SmallDataset, создаем SmallDataset
                raise TypeError(f"Wrong value {value} for converting to SmallDataset")
            self.analysis_tables[executor_id] = value

        # Handle variables
        elif space == ExperimentDataEnum.variables:
            if executor_id in self.variables:
                self.variables[executor_id][key] = value
            elif isinstance(value, dict):
                self.variables[executor_id] = value
            else:
                self.variables[executor_id] = {key: value}

        # Handle groups
        elif space == ExperimentDataEnum.groups:
            if executor_id not in self.groups:
                self.groups[executor_id] = {key: value}
            else:
                self.groups[executor_id][key] = value

        return self

    def get_ids(
        self,
        classes: type | Iterable[type] | str | Iterable[str],
        searched_space: ExperimentDataEnum | Iterable[ExperimentDataEnum] | None = None,
        key: str | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        def check_id(id_: str, class_: str) -> bool:
            result = id_[: id_.find(ID_SPLIT_SYMBOL)] == class_

            if result and key is not None:
                result = id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :] == key
            return result

        # Define spaces to search
        spaces = {
            ExperimentDataEnum.additional_fields: self.additional_fields.columns,
            ExperimentDataEnum.analysis_tables: self.analysis_tables.keys(),
            ExperimentDataEnum.groups: self.groups.keys(),
            ExperimentDataEnum.variables: self.variables.keys(),
        }

        # Convert classes to names
        classes = [
            c.__name__ if isinstance(c, type) else c for c in Adapter.to_list(classes)
        ]

        # Get spaces to search
        searched_space = (
            Adapter.to_list(searched_space) if searched_space else list(spaces.keys())
        )

        # Return matching IDs
        return {
            class_: {
                space.value: [
                    str(id_) for id_ in spaces[space] if check_id(id_, class_)
                ]
                for space in searched_space
            }
            for class_ in classes
        }

    def get_one_id(
        self,
        class_: type | str,
        space: ExperimentDataEnum,
        key: str | None = None,
    ) -> str:
        class_ = class_ if isinstance(class_, str) else class_.__name__
        result = self.get_ids(class_, space, key)
        if (class_ not in result) or (not len(result[class_][space.value])):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

    def copy(self, data: Dataset | None = None) -> ExperimentData:
        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result

    def field_search(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types=None,
    ) -> list[str]:
        searched_field = []
        roles = Adapter.to_list(roles)

        # Split roles by type
        field_in_additional = [
            role for role in roles if isinstance(role, AdditionalRole)
        ]
        field_in_data = [role for role in roles if role not in field_in_additional]

        # Search in main data
        if field_in_data:
            searched_field += self.ds.search_columns(
                field_in_data, tmp_role=tmp_role, search_types=search_types
            )

        # Search in additional fields
        if field_in_additional and isinstance(self, ExperimentData):
            searched_field += self.additional_fields.search_columns(
                field_in_additional, tmp_role=tmp_role, search_types=search_types
            )

        return searched_field

    def field_data_search(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types = None,
    ) -> Dataset:
        searched_data: Dataset = Dataset.create_empty(index=self._data.index,
                                                      backend=self._data.backend_type,
                                                      session=self._data.session)
        roles = Adapter.to_list(roles)

        # Map roles to columns
        roles_columns_map = {
            role: self.field_search(role, tmp_role, search_types) for role in roles
        }
        
        # Build dataset from matching columns
        for role, columns in roles_columns_map.items():
            for column in columns:
                t_data = (
                    self.additional_fields[column]
                    if isinstance(role, AdditionalRole)
                    else self.ds[column]
                )
                searched_data = searched_data.add_column(
                    data=t_data, role={column: role}
                )
        # if not searched_data.is_empty():
        #     searched_data.index = self.ds.index
        return searched_data
    
    def __deepcopy__(self, memo):
        """deepcopy dataset"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k.startswith('_abc_'):
                continue
            setattr(result, k, deepcopy(v, memo))

        return result


class DatasetAdapter(Adapter):
    @staticmethod
    def to_dataset(
        data: dict | Dataset | pd.DataFrame | spark.DataFrame | list | str | int | float | bool,
        roles: ABCRole | dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        # Convert data based on its type
        if isinstance(data, dict):
            return DatasetAdapter.dict_to_dataset(data, roles, small)
        elif (
                isinstance(data, pd.DataFrame) or 
                isinstance(data, spark.DataFrame) or 
                isinstance(data, ps.DataFrame)
            ):
            if isinstance(roles, ABCRole):
                raise InvalidArgumentError("roles", "dict[str, ABCRole]")
            return DatasetAdapter.frame_to_dataset(data, roles, small)
        elif isinstance(data, list):
            if isinstance(roles, ABCRole):
                raise InvalidArgumentError("roles", "dict[str, ABCRole]")
            return DatasetAdapter.list_to_dataset(data, roles, small)
        elif isinstance(data, np.ndarray):
            return DatasetAdapter.ndarray_to_dataset(data, roles, small)
        elif any(isinstance(data, t) for t in [str, int, float, bool]):
            return DatasetAdapter.value_to_dataset(data, roles, small)
        elif isinstance(data, Dataset):
            if small:
                return data.to_small_dataset()
            return data
        elif isinstance(data, SmallDataset):
            if small:
                return data
            return data.to_dataset()
        else:
            raise InvalidArgumentError("data", "dict, pd.DataFrame, list, Dataset")

    @staticmethod
    def value_to_dataset(
        data: ScalarType, roles: ABCRole | dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        if isinstance(roles, ABCRole):
            roles = {"value": roles}
        return Dataset(
            roles=roles, data=pd.DataFrame({next(iter(roles.keys())): [data]})
        )

    @staticmethod
    def dict_to_dataset(
        data: dict, roles: ABCRole | dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        roles_names = list(data.keys())
        if isinstance(roles, dict):
            result = SmallDataset.from_dict(data=data, roles=roles)
        elif isinstance(roles, ABCRole):
            result = SmallDataset.from_dict(
                data=data, roles={name: roles for name in roles_names}
            )
        if not small:
            result = result.to_dataset()
        return result

    @staticmethod
    def list_to_dataset(
        data: list, roles: dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        result = Dataset(
            roles=roles if len(roles) > 0 else {0: DefaultRole()},
            data=pd.DataFrame(
                data=data, columns=[next(iter(roles.keys()))] if len(roles) > 0 else [0]
            ),
        )
        if not small:
            result = result.to_dataset()
        return result

    @staticmethod
    def frame_to_dataset(
        data: pd.DataFrame | spark.DataFrame, 
        roles: dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        if small:
            result = SmallDataset(
                roles=roles,
                data=data,
            )
        else:
            result = Dataset(
                roles=roles,
                data=data,
            )
        return result

    @staticmethod
    def ndarray_to_dataset(
        data: np.ndarray, roles: dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        columns = range(data.shape[1]) if len(roles) == 0 else list(roles.keys())
        data = pd.DataFrame(data=data, columns=columns)
        result = SmallDataset(
            roles=roles,
            data=data,
        )
        if not small:
            result = result.to_dataset()
        return result
