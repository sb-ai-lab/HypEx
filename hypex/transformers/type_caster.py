from __future__ import annotations

from typing import Any, Sequence

from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import ABCRole, FeatureRole
from .abstract import Transformer


class TypeCaster(Transformer):
    def __init__(
        self,
        dtype: dict[str, type] | dict[type, type],
        roles: ABCRole | Sequence[ABCRole] | None = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.dtype = dtype
        self.roles = roles or FeatureRole()

    @staticmethod
    def _inner_function(
        data: Dataset,
        dtype: dict[str, type],
    ) -> Dataset:
        return data.astype(dtype=dtype)

    @classmethod
    def calc(
        cls,
        data: Dataset,
        dtype: dict[str, type] | dict[type, type],
        roles: ABCRole | Sequence[ABCRole] | None = None,
        **kwargs,
    ):
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
            )
        )
        return result
