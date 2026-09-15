from __future__ import annotations

from decimal import Decimal
from collections.abc import Sequence
from typing import Any

from ..dataset.dataset import Dataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import ABCRole, FeatureRole
from .abstract import Transformer


class TypeCaster(Transformer):
    def __init__(
        self,
        dtype: dict[str, type] | dict[type, type],
        roles: ABCRole | Sequence[ABCRole] | None = None,
        downcasting: bool=True,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.dtype = dtype
        self.roles = roles or FeatureRole()
        self.downcasting = downcasting

    @staticmethod
    def _inner_function(
        data: Dataset,
        dtype: dict[str, type],
    ) -> Dataset:
        return data.astype(dtype=dtype)

    @staticmethod
    def _downcast(
        data: Dataset,
    ) -> Dataset:
        double_cols = [
            col for col, c_type in data.roles.items()
            if c_type.data_type is Decimal or c_type.data_type is float
        ]
        return data.astype({col: float for col in double_cols})

    @classmethod
    def calc(
        cls,
        data: Dataset,
        dtype: dict[str, type] | dict[type, type],
        roles: ABCRole | Sequence[ABCRole] | None = None,
        **kwargs,
    ):
        downcasting = kwargs.pop("downcasting", True)
        if downcasting:
            data = TypeCaster._downcast(data=data)

        cast_mapping = {}
        for k, v in dtype.items():
            if isinstance(k, str):
                cast_mapping[k] = v
            elif isinstance(k, type):
                cast_mapping.update({c: v for c in data.search_columns_by_type(k)})
        if roles:
            target_cols = data.search_columns(roles=roles)
            cast_mapping = {c: v for c, v in cast_mapping.items() if c in target_cols}

        return cls._inner_function(data, cast_mapping, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(
                data=data.ds,
                dtype=self.dtype,
                roles=self.roles,
                downcasting=self.downcasting,
            )
        )
        return result
