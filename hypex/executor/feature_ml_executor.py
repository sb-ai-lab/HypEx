from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    GroupingRole,
    TargetRole,
)
from ..utils import (
    ID_SPLIT_SYMBOL,
    AbstractMethodError,
    ExperimentDataEnum,
    NotSuitableFieldError,
)
from ..utils.adapter import Adapter
from .executor import Calculator


class FeatureMLExecutor(Calculator, ABC):
    """Legacy feature-based ML executor used by matching pipelines (e.g. Faiss)."""

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_role: ABCRole | None = None,
        key: Any = "",
    ):
        self.target_role = target_role or TargetRole()
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()

    def _get_fields(self, data: ExperimentData):
        group_field = data.field_search(self.grouping_role)
        target_field = data.field_search(
            self.target_role, search_types=self.search_types
        )
        return group_field, target_field

    @abstractmethod
    def fit(self, X: Dataset, Y: Dataset | None = None) -> FeatureMLExecutor:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        raise NotImplementedError

    def score(self, X: Dataset, Y: Dataset) -> float:
        raise NotImplementedError

    @property
    def search_types(self):
        return [int, float]

    @classmethod
    @abstractmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_data: Dataset | None = None,
        **kwargs,
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_field: str | None = None,
        **kwargs,
    ) -> Any:
        if target_field:
            return cls._inner_function(
                data=grouping_data[0][1].drop(target_field),
                target_data=grouping_data[0][1][target_field],
                test_data=grouping_data[1][1].drop(target_field),
                **kwargs,
            )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            **kwargs,
        )

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        for i in range(value.shape[1]):
            data.set_value(
                ExperimentDataEnum.additional_fields,
                f"{self.id}{ID_SPLIT_SYMBOL}{i}",
                value=value.iloc[:, i],
                key=key,
                role=AdditionalMatchingRole(),
            )
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Sequence[str] | str | None = None,
        grouping_data: list[tuple[str, Dataset]] | None = None,
        target_field: str | list[str] | None = None,
        features_fields: str | list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        group_field = Adapter.to_list(group_field)
        features_fields = Adapter.to_list(features_fields)
        if grouping_data is None:
            grouping_data = data.groupby(group_field, fields_list=features_fields)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise NotSuitableFieldError(group_field, "Grouping")
        return cls._execute_inner_function(
            grouping_data, target_field=target_field, **kwargs
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        features_fields = data.ds.search_columns(
            FeatureRole(), search_types=self.search_types
        )
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if not target_fields and data.ds.tmp_roles:
            return data

        if group_field[0] in data.groups:
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = None

        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            target_fields=target_fields,
            features_fields=features_fields,
        )
        return self._set_value(data, compare_result)
