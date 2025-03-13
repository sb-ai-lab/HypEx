from __future__ import annotations

from typing import Any, Sequence

from ..dataset import Dataset, ExperimentData, FeatureRole
from ..executor import Calculator
from ..utils import (
    NAME_BORDER_SYMBOL,
    AbstractMethodError,
    CategoricalTypes,
    ExperimentDataEnum,
)


class Encoder(Calculator):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        key: Any = "",
    ):
        self.target_roles = target_roles or FeatureRole()
        self._key = key
        super().__init__(key)

    @property
    def __is_encoder(self):
        return True

    @property
    def search_types(self):
        return [CategoricalTypes]

    def _get_ids(self, col_name):
        self.key = f"{NAME_BORDER_SYMBOL}{col_name}{NAME_BORDER_SYMBOL}"
        return self.id

    def _ids_to_names(self, col_names: list[str]):
        return {col_name: self._get_ids(col_name) for col_name in col_names}

    @staticmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        raise AbstractMethodError

    def _set_value(
        self, data: ExperimentData, value: Dataset, key=None
    ) -> ExperimentData:
        return data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id=self._ids_to_names(value.columns),
            value=value,
            role=value.roles,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(
            roles=self.target_roles, search_types=self.search_types
        )
        return self._set_value(
            data=data,
            value=self.calc(data=data.ds, target_cols=target_cols),
            key=self.key,
        )
