from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from ..dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    GroupingRole,
    TargetRole,
)
from ..executor import Calculator
from ..extensions.scipy_linalg import CholeskyExtension, InverseExtension
from ..utils import ExperimentDataEnum, NotSuitableFieldError
from ..utils.adapter import Adapter


class MahalanobisDistance(Calculator):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        result = {}
        for i in range(1, len(grouping_data)):
            result.update(
                cls._inner_function(
                    data=(
                        grouping_data[0][1][target_fields]
                        if target_fields
                        else grouping_data[0][1]
                    ),
                    test_data=(
                        grouping_data[i][1][target_fields]
                        if target_fields
                        else grouping_data[i][1]
                    ),
                    **kwargs,
                )
            )
        return result

    def _set_value(
        self, data: ExperimentData, value: dict | None = None, key: Any = None
    ) -> ExperimentData:
        for key, value_ in value.items():
            data = data.set_value(
                ExperimentDataEnum.groups,
                self.id,
                value_,
                key=key,
            )
        return data

    def _get_fields(self, data: ExperimentData):
        group_field = data.field_search(self.grouping_role)
        target_fields = data.field_search(FeatureRole(), search_types=self.search_types)
        return group_field, target_fields

    @property
    def search_types(self) -> list[type] | None:
        return [int, float]

    @classmethod
    def _inner_function(cls, data: Dataset, test_data: Dataset | None = None, **kwargs):
        test_data = cls._check_test_data(test_data)
        cov = (data.cov() + test_data.cov()) / 2 if test_data else data.cov()
        cholesky = CholeskyExtension().calc(cov)
        mahalanobis_transform = InverseExtension().calc(cholesky)
        y_control = data.dot(mahalanobis_transform.transpose())
        if test_data:
            y_test = test_data.dot(mahalanobis_transform.transpose())
            return {"control": y_control, "test": y_test}
        return {"control": y_control}

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

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = None
        t_data = deepcopy(data.ds)
        if target_fields[1] not in t_data.columns:
            t_data = t_data.add_column(
                data.additional_fields[target_fields[1]],
                role={target_fields[1]: TargetRole()},
            )
        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            grouping_data=grouping_data,
        )
        return self._set_value(data, compare_result)
