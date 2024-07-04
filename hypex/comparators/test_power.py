# from typing import Any
#
# import numpy as np
# from scipy.stats import norm
# from statsmodels.stats.power import TTestIndPower
#
# from hypex.comparators.comparators import ComparatorInner
# from hypex.dataset import ExperimentData
# from hypex.utils import
#
#
#
#
# class TestPower(ComparatorInner):
#     def __init__(
#         self,
#         target_field: ,
#         significance: float = 0.95,
#         power: float = 0.8,
#         full_name: str = None,
#         key: Any = "",
#     ):
#         super().__init__(target_field, self.comparison_function, full_name, key)
#         self.significance = significance
#         self.power = power
#
#
# class StatMdeBySize(TestPower):
#     def _inner_function(self, control_data, test_data) -> ExperimentData:
#         m = norm.ppf(1 - self.significance / 2) + norm.ppf(self.power)
#
#         n_test, n_control = len(test_data), len(control_data)
#         proportion = n_test / (n_test + n_control)
#         p = np.sqrt(1 / (proportion * (1 - proportion)))
#
#         var_test, var_control = np.var(test_data, ddof=1), np.var(control_data, ddof=1)
#         s = np.sqrt(var_test / n_test + var_control / n_control)
#
#         return p * m * s
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
from typing import Optional, Any, Dict, Union, Sequence

import numpy as np
from scipy.stats import norm

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import ABCRole, Dataset, ExperimentData, TargetRole
from hypex.utils import SpaceEnum


class MDEBySize(GroupComparator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        full_name: Optional[str] = None,
        key: Any = "",
        power: float = 0.8,
        significance: float = 0.95,
    ):
        super().__init__(grouping_role, space, full_name, key)
        self.power = power
        self.significance = significance

    @staticmethod
    def _inner_function(
        control_data, test_data, significance=0.95, power=0.8, **kwargs
    ) -> Dict[str, Any]:
        result = {}
        m = norm.ppf(1 - significance / 2) - norm.ppf(power)
        n_control, n_test = len(control_data), len(test_data)
        proportion = n_test / (n_test + n_control)
        p = np.sqrt(1 / (proportion * (1 - proportion)))
        for target in control_data.columns:
            var_control = control_data[target].var()
            var_test = test_data[target].var()
            s = np.sqrt(var_test / n_test + var_control / n_control)
            result[target] = p * m * s

        return result

    @staticmethod
    def calc(
        cls: Dataset,
        data: Union[Sequence[str], str, None],
        group_field: Optional[str] = None,
        grouping_data=None,
        target_fields=None,
        **kwargs
    ):
        return GroupComparator.calc(
            data=data,
            group_field=group_field,
            target_fields=target_fields,
            comparison_function=MDEBySize._inner_function,
            power=power,
            significance=target_fields,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        subdata = data.ds.loc[
            :, data.ds.get_columns_by_roles([TargetRole(), self.grouping_role])
        ]
        ed = super().execute(ExperimentData(subdata))
        return self._set_value(data, ed.analysis_tables[self._id])
