# from typing import Any
#
# import numpy as np
# from scipy.stats import norm
# from statsmodels.stats.power import TTestIndPower
#
# from hypex.comparators.comparators import ComparatorInner
# from hypex.dataset import ExperimentData
# from hypex.utils import FieldKeyTypes
#
#
# # TODO: Rework ALL
#
#
# class TestPower(ComparatorInner):
#     def __init__(
#         self,
#         target_field: FieldKeyTypes,
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
#     def _comparison_function(self, control_data, test_data) -> ExperimentData:
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
# # TODO: replace target_field on subroles
# class StatSampleSizeByMde(TestPower):
#     def __init__(
#         self,
#         mde: float,
#         target_field: FieldKeyTypes,
#         significance: float = 0.05,
#         power: float = 0.8,
#         full_name: str = None,
#         key: Any = "",
#     ):
#         super().__init__(target_field, significance, power, full_name, key)
#         self.mde = mde
#
#     # TODO: rework by ExperimentData checking
#     def _comparison_function(self, control_data, test_data) -> ExperimentData:
#         control_std = control_data.std()
#         test_std = test_data.std()
#
#         test_proportion = len(test_data) / (len(test_data) + len(control_data))
#         control_proportion = 1 - test_proportion
#
#         d = ((norm.ppf(1 - self.significance / 2) + norm.ppf(power)) / self.mde) ** 2
#         s = test_std**2 / test_proportion + control_std**2 / control_proportion
#         return int(d * s)
#
#
# class StatPowerByTTestInd(TestPower):
#
#     def _comparison_function(self, control_data, test_data) -> ExperimentData:
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
#         )
