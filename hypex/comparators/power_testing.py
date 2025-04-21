from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import norm

from ..dataset import ABCRole, Dataset, ExperimentData
from ..utils import ExperimentDataEnum
from .comparators import Comparator


class PowerTesting(Comparator, ABC):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        # space: SpaceEnum = SpaceEnum.auto,
        significance: float = 0.95,
        power: float = 0.8,
        key: Any = "",
    ):
        super().__init__(
            compare_by="groups",
            grouping_role=grouping_role,
            # space=space,
            key=key,
        )
        self.significance = significance
        self.power = power

    @classmethod
    @abstractmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        significance: float = 0.95,
        power: float = 0.8,
        **kwargs,
    ) -> float:
        pass

    def execute(self, data: ExperimentData) -> ExperimentData:
        return super().execute(data)


class MDEBySize(PowerTesting):
    def _set_value(
        self, data: ExperimentData, value: Dataset | None = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            value,
        )
        return data

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        significance: float = 0.95,
        power: float = 0.8,
        **kwargs,
    ) -> float:
        m = norm.ppf((1 + significance) / 2) + norm.ppf(power)
        if not test_data:
            raise ValueError("test_data is required")

        n_test, n_control = len(test_data), len(data)

        var_test, var_control = test_data.var(ddof=1), data.var(ddof=1)
        s = np.sqrt(var_test / n_test + var_control / n_control)

        return m * s


#
#
# class StatPowerByTTestInd(TestPower):
#
#     def _inner_function(self, control_data, test_data) -> ExperimentData:
#         control_size = len(control_data)
#         test_size = len(test_data)
#
#         analysis = TTestIndPower()
#         ratio = test_size / control_size
#         return analysis.power(
#             effect_size=effect_size,
#             nobs1=test_size,
#             ratio=ratio,
#             alpha=significance,
#


# class MDEBySize(GroupComparator):
#     def __init__(
#         self,
#         grouping_role: Optional[ABCRole] = None,
#         space: SpaceEnum = SpaceEnum.auto,
#         full_name: Optional[str] = None,
#         key: Any = "",
#         power: float = 0.8,
#         significance: float = 0.95,
#     ):
#         super().__init__(grouping_role, space, full_name, key)
#         self.power = power
#         self.significance = significance
#
#     @staticmethod
#     def _inner_function(
#         control_data, test_data, significance=0.95, power=0.8, **kwargs
#     ) -> Dict[str, Any]:
#         result = {}
#         m = norm.ppf(1 - significance / 2) - norm.ppf(power)
#         n_control, n_test = len(control_data), len(test_data)
#         proportion = n_test / (n_test + n_control)
#         p = np.sqrt(1 / (proportion * (1 - proportion)))
#         for target in control_data.columns:
#             var_control = control_data[target].var()
#             var_test = test_data[target].var()
#             s = np.sqrt(var_test / n_test + var_control / n_control)
#             result[target] = p * m * s
#
#         return result
#
#     @staticmethod
#     def calc(
#         cls: Dataset,
#         data: Union[Sequence[str], str, None],
#         group_field: Optional[str] = None,
#         grouping_data=None,
#         target_fields=None,
#         **kwargs
#     ):
#         return GroupComparator.calc(
#             data=data,
#             group_field=group_field,
#             target_fields=target_fields,
#             comparison_function=MDEBySize._inner_function,
#             power=power,
#             significance=target_fields,
#         )
#
#     def execute(self, data: ExperimentData) -> ExperimentData:
#         subdata = data.ds.loc[
#             :, data.ds.get_columns_by_roles([TargetRole(), self.grouping_role])
#         ]
#         ed = super().execute(ExperimentData(subdata))
#         return self._set_value(data, ed.analysis_tables[self._id])
