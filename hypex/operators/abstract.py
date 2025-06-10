from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

from ..dataset import (
    ABCRole,
    AdditionalTargetRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    TargetRole,
)
from ..executor import Calculator
from ..utils import AbstractMethodError, ExperimentDataEnum, NotSuitableFieldError
from ..utils.adapter import Adapter


class GroupOperator(
    Calculator
):  # TODO: change the derive from Calculator to COmparator
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.target_roles = target_roles or TargetRole()
        self.grouping_role = grouping_role or GroupingRole()

    @property
    def search_types(self):
        return None

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    def _get_fields(self, data: ExperimentData):
        group_field = data.field_search(self.grouping_role)
        target_fields = data.field_search(
            self.target_roles, search_types=self.search_types
        )
        if len(target_fields) != 2:
            target_fields += data.field_search(
                AdditionalTargetRole(), search_types=self.search_types
            )
        return group_field, target_fields

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                f"This operator works with 2 targets, but got {len(target_fields) if target_fields else None}"
            )
        result = {}
        for group, group_data in grouping_data:
            result[group[0]] = cls._inner_function(
                data=group_data[target_fields[0]],
                test_data=group_data[target_fields[1]],
                **kwargs,
            )
        return result

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Sequence[str] | str | None = None,
        grouping_data: list[tuple[str, Dataset]] | None = None,
        target_fields: str | list[str] | None = None,
        **kwargs,
    ) -> dict:
        group_field = Adapter.to_list(group_field)

        if grouping_data is None:
            grouping_data = data.groupby(group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise NotSuitableFieldError(group_field, "Grouping")
        return cls._execute_inner_function(
            grouping_data, target_fields=target_fields, old_data=data, **kwargs
        )

    def _set_value(
        self, data: ExperimentData, value: dict | None = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            value,
        )
        return data
