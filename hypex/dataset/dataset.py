from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import pyspark.sql as spark

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
        data_compression: Literal[
            "downcasting", "encoding", "auto", "disable"
        ] = "auto",
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
        backend: BackendsEnum | None = None,
    ):
        if isinstance(roles, dict) and data is not None:
            columns = None

            if hasattr(data, "columns"):
                cols = data.columns
                columns = cols.tolist() if hasattr(cols, "tolist") else list(cols)
            elif hasattr(data, "name") and data.name is not None:
                columns = [data.name]
            elif (
                isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], dict)
            ):
                columns = list(data["data"].keys())

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
        self.loc = self.Locker(
            call_class=self.__class__, backend=self._backend_data, roles=self.roles
        )
        self.iloc = self.ILocker(
            call_class=self.__class__, backend=self._backend_data, roles=self.roles
        )

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
                roles = {col: roles for col in data.keys()}
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


class DatasetAdapter(Adapter):
    @staticmethod
    def to_dataset(
        data: dict | Dataset | pd.DataFrame | list | str | int | float | bool,
        roles: ABCRole | dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        # Convert data based on its type
        if isinstance(data, dict):
            return DatasetAdapter.dict_to_dataset(data, roles, small)
        elif isinstance(data, pd.DataFrame):
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
        data: ScalarType,
        roles: ABCRole | dict[str, ABCRole],
        small: bool = True,
    ) -> Dataset | SmallDataset:
        if isinstance(roles, ABCRole):
            roles = {"value": roles}
        return Dataset(
            roles=roles, data=pd.DataFrame({next(iter(roles.keys())): [data]})
        )

    @staticmethod
    def dict_to_dataset(
        data: dict,
        roles: ABCRole | dict[str, ABCRole],
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
        data: list,
        roles: dict[str, ABCRole],
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
        data: pd.DataFrame,
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
        data: np.ndarray,
        roles: dict[str, ABCRole],
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
